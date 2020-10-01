# ActivitySim
# See full license in LICENSE.txt.
from builtins import range

import logging
import time
import math

from contextlib import contextmanager

import numpy as np
import pandas as pd

from activitysim.core import tracing
from activitysim.core import inject
from activitysim.core import config
from activitysim.core import chunk
from activitysim.core import logit
from activitysim.core import simulate
from activitysim.core import los

from activitysim.core.util import reindex

from activitysim.core import expressions
from activitysim.core import assign

logger = logging.getLogger(__name__)

TIMING = True
TRACE_CHUNK = True


@contextmanager
def timeit(trace_label, tag=None):
    t0 = tracing.print_elapsed_time()
    try:
        yield
    finally:
        t = time.time() - t0
    if TIMING:
        logger.debug(f"TVPB TIME {trace_label} {tag} {tracing.format_elapsed_time(t)}")


def chunk_log_df(trace_label, name, df):

    chunk.log_df(trace_label, name, df)

    if TRACE_CHUNK:
        logger.debug(f"TVPB {name} {df.shape}")


def trace_maz_tap(maz_od_df, access_mode, egress_mode, network_los):

    def maz_tap_stats(mode, name):
        maz_tap_df = network_los.maz_to_tap_dfs[mode].reset_index()
        logger.debug(f"TVPB access_maz_tap {maz_tap_df.shape}")
        MAZ_count = len(maz_tap_df.MAZ.unique())
        TAP_count = len(maz_tap_df.TAP.unique())
        MAZ_PER_TAP = MAZ_count / TAP_count
        logger.debug(f"TVPB maz_tap_stats {name} {mode} MAZ {MAZ_count} TAP {TAP_count} ratio {MAZ_PER_TAP}")

    logger.debug(f"TVPB maz_od_df {maz_od_df.shape}")

    maz_tap_stats(access_mode, 'access')
    maz_tap_stats(egress_mode, 'egress')


class TransitVirtualPathBuilder(object):

    def __init__(self, network_los):

        self.network_los = network_los

        # FIXME - need to recompute headroom?
        self.chunk_size = inject.get_injectable('chunk_size')

        assert network_los.zone_system == los.THREE_ZONE, \
            f"TransitVirtualPathBuilder: network_los zone_system not THREE_ZONE"

    def trace_df(self, df, trace_label, extension=None, bug=False):

        if extension:
            trace_label = tracing.extend_trace_label(trace_label, extension)

        assert len(df) > 0

        tracing.trace_df(df, label=trace_label, slicer='NONE', transpose=False)

        if bug:
            print(f"{trace_label}\n{df}")
            bug_out

    def compute_utilities(self, model_settings, choosers, spec, locals_dict, network_los, trace_label, trace):

        trace_label = tracing.extend_trace_label(trace_label, 'compute_utilities')

        logger.debug(f"{trace_label} Running compute_utilities with {choosers.shape[0]} choosers")

        locals_dict = locals_dict.copy()  # don't clobber argument
        locals_dict.update({
            'np': np,
            'los': network_los
        })

        # - run preprocessor to annotate choosers
        preprocessor_settings = model_settings.get('PREPROCESSOR')
        if preprocessor_settings:

            # don't want to alter caller's dataframe
            choosers = choosers.copy()

            expressions.assign_columns(
                df=choosers,
                model_settings=preprocessor_settings,
                locals_dict=locals_dict,
                trace_label=trace_label)

            if trace:
                self.trace_df(choosers, trace_label, 'choosers')

        utilities = simulate.eval_utilities(
            spec,
            choosers,
            locals_d=locals_dict,
            trace_all_rows=trace,
            trace_label=trace_label)

        return utilities

    def compute_maz_tap_utilities(self, recipe, maz_od_df, chooser_attributes, leg, mode, trace_label, trace):

        trace_label = tracing.extend_trace_label(trace_label, f'compute_maz_tap_utilities.{leg}')

        maz_tap_settings = \
            self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.maz_tap_expressions.{mode}')
        chooser_columns = maz_tap_settings['CHOOSER_COLUMNS']
        attribute_columns = list(chooser_attributes.columns) if chooser_attributes is not None else []
        model_constants = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.CONSTANTS')
        units = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.units')

        if leg == 'access':
            maz_col = 'omaz'
            tap_col = 'btap'
        else:
            maz_col = 'dmaz'
            tap_col = 'atap'

        # maz_to_tap access/egress
        # deduped access_df - one row per chooser for each boarding tap (btap) accessible from omaz
        access_df = self.network_los.maz_to_tap_dfs[mode]

        access_df = access_df[chooser_columns]. \
            reset_index(drop=False). \
            rename(columns={'MAZ': maz_col, 'TAP': tap_col})
        access_df = pd.merge(
            maz_od_df[['idx', maz_col]].drop_duplicates(),
            access_df,
            on=maz_col, how='inner')
        # add any supplemental chooser attributes (e.g. demographic_segment, tod)
        for c in attribute_columns:
            access_df[c] = reindex(chooser_attributes[c], access_df['idx'])

        if units == 'utility':

            maz_tap_spec = simulate.read_model_spec(file_name=maz_tap_settings['SPEC'])

            access_df[leg] = self.compute_utilities(
                maz_tap_settings,
                access_df,
                spec=maz_tap_spec,
                locals_dict=model_constants,
                network_los=self.network_los,
                trace_label=trace_label, trace=trace)

        else:

            assignment_spec = assign.read_assignment_spec(file_name=config.config_file_path(maz_tap_settings['SPEC']))

            results, _, _ = assign.assign_variables(assignment_spec, access_df, model_constants)
            assert len(results.columns == 1)
            access_df[leg] = results

        # drop utility computation columns ('tod', 'demographic_segment' and maz_to_tap_df time/distance columns)
        access_df.drop(columns=attribute_columns + chooser_columns, inplace=True)

        if trace:
            self.trace_df(access_df, trace_label, 'access_df')

        return access_df

    def compute_tap_tap_utilities(self, recipe, access_df, egress_df, chooser_attributes, path_info,
                                  trace_label, trace):

        trace_label = tracing.extend_trace_label(trace_label, 'compute_tap_tap_utilities')

        model_constants = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.CONSTANTS')
        tap_tap_settings = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.tap_tap_expressions')
        attribute_columns = list(chooser_attributes.columns) if chooser_attributes is not None else []
        units = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.units')

        # FIXME some expressions may want to know access mode -
        locals_dict = path_info.copy()
        locals_dict.update(model_constants)

        # compute tap_to_tap utilities
        # deduped transit_df has one row per chooser for each boarding (btap) and alighting (atap) pair
        transit_df = pd.merge(
            access_df[['idx', 'btap']],
            egress_df[['idx', 'atap']],
            on='idx').drop_duplicates()

        # don't want transit trips that start and stop in same tap
        transit_df = transit_df[transit_df.atap != transit_df.btap]

        for c in list(attribute_columns):
            transit_df[c] = reindex(chooser_attributes[c], transit_df['idx'])

        if units == 'utility':
            spec = simulate.read_model_spec(file_name=tap_tap_settings['SPEC'])

            transit_utilities = self.compute_utilities(
                tap_tap_settings,
                choosers=transit_df,
                spec=spec,
                locals_dict=locals_dict,
                network_los=self.network_los,
                trace_label=trace_label, trace=trace)

            transit_df = pd.concat([transit_df[['idx', 'btap', 'atap']], transit_utilities], axis=1)

        else:

            locals_d = {'los': self.network_los}
            locals_d.update(model_constants)

            assignment_spec = assign.read_assignment_spec(file_name=config.config_file_path(tap_tap_settings['SPEC']))

            results, _, _ = assign.assign_variables(assignment_spec, transit_df, locals_d)
            assert len(results.columns == 1)
            transit_df['transit'] = results

            # filter out unavailable btap_atap pairs
            logger.debug(f"{(transit_df['transit'] <= 0).sum()} unavailable tap_tap pairs out of {len(transit_df)}")
            transit_df = transit_df[transit_df.transit > 0]

            transit_df.drop(columns=attribute_columns, inplace=True)

        if trace:
            self.trace_df(transit_df, trace_label, 'transit_df')

        return transit_df

    def best_paths(self, recipe, path_type, maz_od_df, access_df, egress_df, transit_df, trace_label, trace=False):

        trace_label = tracing.extend_trace_label(trace_label, 'best_paths')

        units = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.units')
        transit_sets = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.tap_tap_expressions.sets')

        path_settings = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.path_types.{path_type}')
        max_paths_per_tap_set = path_settings.get('max_paths_per_tap_set', 1)
        max_paths_across_tap_sets = path_settings.get('max_paths_across_tap_sets', 1)

        assert units in ['utility', 'time'], f"unrecognized units: {units}. Expected either 'time' or 'utility'."
        smaller_is_better = (units in ['time'])

        maz_od_df['seq'] = maz_od_df.index
        # maz_od_df has one row per chooser
        # inner join to add rows for each access, egress, and transit segment combination
        path_df = maz_od_df. \
            merge(access_df, on=['idx', 'omaz'], how='inner'). \
            merge(egress_df, on=['idx', 'dmaz'], how='inner'). \
            merge(transit_df, on=['idx', 'atap', 'btap'], how='inner')

        chunk.log_df(trace_label, "path_df", path_df)

        for c in transit_sets:
            path_df[c] = path_df[c] + path_df['access'] + path_df['egress']
        path_df.drop(columns=['access', 'egress'], inplace=True)

        # choose best paths by tap set
        best_paths_list = []
        for c in transit_sets:
            keep = path_df.index.isin(
                path_df[['seq', c]].sort_values(by=c, ascending=smaller_is_better).
                groupby(['seq']).head(max_paths_per_tap_set).index
            )

            best_paths_for_set = path_df[keep]
            best_paths_for_set['path_set'] = c  # remember the path set
            best_paths_for_set[units] = path_df[keep][c]
            best_paths_for_set.drop(columns=transit_sets, inplace=True)
            best_paths_list.append(best_paths_for_set)

        path_df = pd.concat(best_paths_list).sort_values(by=['seq', units], ascending=[True, smaller_is_better])

        # choose best paths overall by seq
        path_df = path_df.sort_values(by=['seq', units], ascending=[True, smaller_is_better])
        path_df = path_df[path_df.index.isin(path_df.groupby(['seq']).head(max_paths_across_tap_sets).index)]

        if trace:
            self.trace_df(path_df, trace_label, 'best_paths.best_paths')

        return path_df

    def build_virtual_path(self, recipe, path_type, orig, dest, tod, demographic_segment,
                           want_choices, trace_label,
                           filter_targets=None, trace=False, override_choices=None):

        trace_label = tracing.extend_trace_label(trace_label, 'build_virtual_path')

        # Tracing is implemented as a seperate, second call that operates ONLY on filter_targets
        assert not (trace and filter_targets is None)
        if filter_targets is not None:
            assert filter_targets.any()

            # slice orig and dest
            orig = orig[filter_targets]
            dest = dest[filter_targets]
            assert len(orig) > 0
            assert len(dest) > 0

            # slice tod and demographic_segment if not scalar
            if not isinstance(tod, str):
                tod = tod[filter_targets]
            if demographic_segment is not None:
                demographic_segment = demographic_segment[filter_targets]
                assert len(demographic_segment) > 0

            # slice choices
            # (requires actual choices from the previous call lest rands change on second call)
            assert want_choices == (override_choices is not None)
            if want_choices:
                override_choices = override_choices[filter_targets]

        units = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.units')
        assert units in ['utility', 'time'], f"unrecognized units: {units}. Expected either 'time' or 'utility'."
        assert units == 'utility' or not want_choices, "'want_choices' only supported supported if units is utility"

        access_mode = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.path_types.{path_type}.access')
        egress_mode = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.path_types.{path_type}.egress')

        # maz od pairs requested
        maz_od_df = pd.DataFrame({
            'idx': orig.index.values,
            'omaz': orig.values,
            'dmaz': dest.values,
            'seq': range(len(orig))
        })
        chunk.log_df(trace_label, "maz_od_df", maz_od_df)
        trace_maz_tap(maz_od_df, access_mode, egress_mode, self.network_los)

        # for location choice, there will be multiple alt dest rows per chooser and duplicate orig.index values
        # but tod and demographic_segment should be the same for all chooser rows (unique orig index values)
        # knowing this allows us to eliminate redundant computations (e.g. utilities of maz_tap pairs)
        duplicated = orig.index.duplicated(keep='first')
        chooser_attributes = pd.DataFrame(index=orig.index[~duplicated])
        chooser_attributes['tod'] = tod if isinstance(tod, str) else tod.loc[~duplicated]
        if demographic_segment is not None:
            chooser_attributes['demographic_segment'] = demographic_segment.loc[~duplicated]

        with timeit(trace_label, "compute_maz_tap_utilities access_df"):
            access_df = self.compute_maz_tap_utilities(
                recipe,
                maz_od_df,
                chooser_attributes,
                leg='access',
                mode=access_mode,
                trace_label=trace_label, trace=trace)
            chunk.log_df(trace_label, "access_df", access_df)

        with timeit(trace_label, "compute_maz_tap_utilities access_df"):
            egress_df = self.compute_maz_tap_utilities(
                recipe,
                maz_od_df,
                chooser_attributes,
                leg='egress',
                mode=egress_mode,
                trace_label=trace_label, trace=trace)
            chunk.log_df(trace_label, "egress_df", egress_df)

        # path_info for use by expressions (e.g. penalty for drive access if no parking at access tap)
        with timeit(trace_label, "compute_tap_tap_utilities"):
            path_info = {'path_type': path_type, 'access_mode': access_mode, 'egress_mode': egress_mode}
            transit_df = self.compute_tap_tap_utilities(
                recipe,
                access_df,
                egress_df,
                chooser_attributes,
                path_info=path_info,
                trace_label=trace_label, trace=trace)
            chunk.log_df(trace_label, "transit_df", transit_df)

        with timeit(trace_label, "compute_tap_tap_utilities"):
            path_df = self.best_paths(recipe, path_type, maz_od_df, access_df, egress_df, transit_df,
                                      trace_label, trace)
            chunk.log_df(trace_label, "path_df", path_df)

        # now that we have created path_df, we are done with the dataframes for the seperate legs
        del access_df
        chunk.log_df(trace_label, "access_df", None)
        del egress_df
        chunk.log_df(trace_label, "egress_df", None)
        del transit_df
        chunk.log_df(trace_label, "transit_df", None)

        if units == 'utility':
            # logsums
            # one row per seq with utilities in columns
            # path_num 0-based to aligh with logit.make_choices 0-based choice indexes
            path_df['path_num'] = path_df.groupby('seq').cumcount()
            chunk.log_df(trace_label, "path_df", path_df)

            utilities_df = path_df[['seq', 'path_num', units]].set_index(['seq', 'path_num']).unstack()
            utilities_df.columns = utilities_df.columns.droplevel()  # for legibility
            chunk.log_df(trace_label, "utilities_df", utilities_df)

            # paths with fewer than the max number of paths will have Nan values for missing data
            # but there should always be at least one path/utility per seq
            # FIXME what if there is no tap-tap transit availability?
            assert not utilities_df.isnull().all(axis=1).any()

            assert (utilities_df.index == maz_od_df.seq).all()  # should be aligned with maz_od_df and so with orig

            # logsum of utilities_df columns
            logsum = np.log(np.nansum(np.exp(utilities_df.values), axis=1))

            if want_choices:

                # utilities for missing paths will be Nan
                utilities_df = utilities_df.fillna(-999.0)
                # orig index to identify appropriate random number channel to use making choices
                utilities_df.index = orig.index

                with timeit(trace_label, "make_choices"):
                    probs = logit.utils_to_probs(utilities_df, trace_label=trace_label)
                    chunk.log_df(trace_label, "probs", probs)

                    if trace:
                        choices = override_choices

                        utilities_df['choices'] = choices
                        self.trace_df(utilities_df, trace_label, 'utilities_df')

                        probs['choices'] = choices
                        self.trace_df(probs, trace_label, 'probs')
                    else:
                        choices, rands = logit.make_choices(probs, trace_label=trace_label)

                    del utilities_df
                    chunk.log_df(trace_label, "utilities_df", None)

                    del probs
                    chunk.log_df(trace_label, "probs", None)

                    # we need to get btap and atap from path_df with same seq and path_num
                    columns_to_cache = ['btap', 'atap', 'path_set']
                    logsum_df = \
                        pd.merge(pd.DataFrame({'seq': range(len(orig)), 'path_num': choices.values}),
                                 path_df[['seq', 'path_num'] + columns_to_cache],
                                 on=['seq', 'path_num'], how='left')

                    # keep path_num choice for caller to pass as override_choices when tracing
                    logsum_df.drop(columns=['seq'], inplace=True)

                    logsum_df.index = orig.index
                    logsum_df['logsum'] = logsum

            else:

                assert len(logsum) == len(orig)
                logsum_df = pd.DataFrame({'logsum': logsum}, index=orig.index)

            if trace:
                self.trace_df(logsum_df, trace_label, 'logsum_df')

            chunk.log_df(trace_label, "logsum_df", logsum_df)
            results = logsum_df

        elif units == 'time':

            # return a series
            results = pd.Series(path_df[units].values, index=path_df['idx'])

            # zero-fill rows for O-D pairs where no best path exists because there was no tap-tap transit availability
            results = reindex(results, maz_od_df.idx).fillna(0.0)

            chunk.log_df(trace_label, "results", results)

        else:
            raise RuntimeError(f"Unrecognized units: '{units}")

        assert len(results) == len(orig)

        del path_df
        chunk.log_df(trace_label, "path_df", None)

        # diagnostic
        # maz_od_df['DIST'] = self.network_los.get_default_skim_dict().get('DIST').get(maz_od_df.omaz, maz_od_df.dmaz)
        # maz_od_df[units] = results.logsum if units == 'utility' else results.values
        # print(f"maz_od_df\n{maz_od_df}")

        return results

    def get_logsum_chunk_overhead(self, orig, dest, tod, demographic_segment, want_choices, trace_label):

        trace_label = tracing.extend_trace_label(trace_label, 'get_logsum_chunk_overhead')

        recipe = 'tour_mode_choice'
        path_types = self.network_los.setting(f'TRANSIT_VIRTUAL_PATH_SETTINGS.{recipe}.path_types').keys()

        # create a boolean array the length of orig with sample_size true values
        CHUNK_OVERHEAD_SAMPLE_SIZE = 50
        sample_size = min(CHUNK_OVERHEAD_SAMPLE_SIZE, len(orig))
        filter_targets = np.zeros(len(orig), dtype=bool)
        filter_targets[np.random.choice(len(orig), size=sample_size, replace=False)] = True
        filter_targets = pd.Series(filter_targets, index=orig.index)

        logger.info(f"{trace_label} running sample size of {sample_size} for path_types: {path_types}")

        # doesn't make any difference what they choose
        override_choices = pd.Series(0, index=orig.index) if want_choices else None

        oh = {}
        for path_type in path_types:

            # chunker will track our memory footprint
            with chunk.chunk_log(trace_label):

                # run asample
                self.build_virtual_path(recipe, path_type, orig, dest, tod, demographic_segment,
                                        want_choices=want_choices,
                                        override_choices=override_choices,
                                        trace_label=trace_label,
                                        filter_targets=filter_targets,
                                        trace=False
                                        )

                # get number of elements allocated during this chunk from the high water mark dict
                hwm_elements = chunk.get_high_water_mark().get('elements').get('mark')

            row_size = math.ceil(hwm_elements / sample_size)

            oh[path_type] = row_size

        return oh

    def get_tvpb_logsum(self, path_type, orig, dest, tod, demographic_segment, want_choices, trace_label=None):

        # assume they have given us a more specific name (since there may be more than one active wrapper)
        trace_label = trace_label or 'get_tvpb_logsum'
        trace_label = tracing.extend_trace_label(trace_label, path_type)

        recipe = 'tour_mode_choice'

        with chunk.chunk_log(trace_label):
            logsum_df = \
                self.build_virtual_path(recipe, path_type, orig, dest, tod, demographic_segment,
                                        want_choices=want_choices, trace_label=trace_label)

            # log number of elements allocated during this chunk from the high water mark dict
            # oh = chunk.get_high_water_mark().get('elements').get('mark')
            # row_size = math.ceil(oh / len(orig))
            # logger.debug(f"#chunk_history get_tvpb_logsum {trace_label} oh: {oh} row_size: {row_size}")

            trace_hh_id = inject.get_injectable("trace_hh_id", None)
            if trace_hh_id:
                filter_targets = tracing.trace_targets(orig)
                # choices from preceding run (because random numbers)
                override_choices = logsum_df['path_num'] if want_choices else None
                if filter_targets.any():
                    self.build_virtual_path(recipe, path_type, orig, dest, tod, demographic_segment,
                                            want_choices=want_choices, override_choices=override_choices,
                                            trace_label=trace_label, filter_targets=filter_targets, trace=True)

        return logsum_df

    def get_tvpb_best_transit_time(self, orig, dest, tod):

        # FIXME lots of pathological knowledge here as we are only called by accessibility directly from expressions

        trace_label = tracing.extend_trace_label('accessibility.get_tvpb_best_transit_time', tod)
        recipe = 'accessibility'
        path_type = 'WTW'

        with chunk.chunk_log(trace_label):
            result = \
                self.build_virtual_path(recipe, path_type, orig, dest, tod,
                                        demographic_segment=None, want_choices=False,
                                        trace_label=trace_label)

            trace_od = inject.get_injectable("trace_od", None)
            if trace_od:
                filter_targets = (orig == trace_od[0]) & (dest == trace_od[1])
                if filter_targets.any():
                    self.build_virtual_path(recipe, path_type, orig, dest, tod,
                                            demographic_segment=None, want_choices=False,
                                            trace_label=trace_label, filter_targets=filter_targets, trace=True)

        return result

    def wrap_logsum(self, orig_key, dest_key, tod_key, segment_key,
                    cache_choices=False, trace_label=None, tag=None):

        return TransitVirtualPathLogsumWrapper(self, orig_key, dest_key, tod_key, segment_key,
                                               cache_choices, trace_label, tag)


class TransitVirtualPathLogsumWrapper(object):

    def __init__(self, transit_virtual_path_builder, orig_key, dest_key, tod_key, segment_key,
                 cache_choices, trace_label, tag):

        self.tvpb = transit_virtual_path_builder
        assert hasattr(transit_virtual_path_builder, 'get_tvpb_logsum')

        self.orig_key = orig_key
        self.dest_key = dest_key
        self.tod_key = tod_key
        self.segment_key = segment_key
        self.df = None

        self.cache_choices = cache_choices
        self.cache = {} if cache_choices else None

        self.base_trace_label = tracing.extend_trace_label(trace_label, tag) or f'tvpb_logsum.{tag}'
        self.trace_label = self.base_trace_label
        self.tag = tag

        self.chunk_overhead = None

        assert isinstance(orig_key, str)
        assert isinstance(dest_key, str)
        assert isinstance(tod_key, str)
        assert isinstance(segment_key, str)

    def set_df(self, df):
        """
        Set the dataframe

        Parameters
        ----------
        df : DataFrame
            The dataframe which contains the origin and destination ids

        Returns
        -------
        self (to facilitiate chaining)
        """

        self.df = df
        return self

    def extend_trace_label(self, extension=None):
        if extension:
            self.trace_label = tracing.extend_trace_label(self.base_trace_label, extension)
        else:
            self.trace_label = self.base_trace_label

    def __getitem__(self, path_type):
        """
        Get an available skim object

        Parameters
        ----------
        key : hashable
             The key (identifier) for this skim object

        Returns
        -------
        skim: Skim
             The skim object
        """

        assert self.df is not None, "Call set_df first"
        assert(self.orig_key in self.df), \
            f"TransitVirtualPathLogsumWrapper: orig_key '{self.orig_key}' not in df"
        assert(self.dest_key in self.df), \
            f"TransitVirtualPathLogsumWrapper: dest_key '{self.dest_key}' not in df"
        assert(self.tod_key in self.df), \
            f"TransitVirtualPathLogsumWrapper: tod_key '{self.tod_key}' not in df"
        assert(self.segment_key in self.df), \
            f"TransitVirtualPathLogsumWrapper: segment_key '{self.segment_key}' not in df"

        orig = self.df[self.orig_key].astype('int')
        dest = self.df[self.dest_key].astype('int')
        tod = self.df[self.tod_key]
        segment = self.df[self.segment_key]

        logsum_df, oh = \
            self.tvpb.get_tvpb_logsum(path_type, orig, dest, tod, segment,
                                      want_choices=self.cache_choices,
                                      trace_label=self.trace_label)

        if self.cache_choices:

            # not tested on duplicate index because not currently needed
            # caching strategy does not require unique indexes but care would need to be taken to maintain alignment
            assert not orig.index.duplicated().any()

            # we only need to cache taps and path_set
            choices_df = logsum_df[['atap', 'btap', 'path_set']]

            if path_type in self.cache:
                assert len(self.cache.get(path_type).index.intersection(logsum_df.index)) == 0
                choices_df = pd.concat([self.cache.get(path_type), choices_df])

            self.cache[path_type] = choices_df

        return logsum_df.logsum

    def estimate_overhead(self, df, trace_label):

        # not really any of our business but...
        # would expect to be called during chunk estimation BEFORE df is set
        assert self.df is None
        assert chunk.not_chunking()

        trace_label = tracing.extend_trace_label(trace_label, 'estimate_overhead')

        orig = df[self.orig_key].astype('int')
        dest = df[self.dest_key].astype('int')
        tod = df[self.tod_key]
        segment = df[self.segment_key]

        oh = self.tvpb.get_logsum_chunk_overhead(orig, dest, tod, segment,
                                                 want_choices=self.cache_choices,
                                                 trace_label=trace_label)

        for path_type, row_size in oh.items():
            logger.info(f"{trace_label} tag {self.tag} path_type {path_type} row_size {row_size} ")

        # spare them the details
        max_row_size = max(oh.values())

        if self.cache_choices:
            # room to cache 'atap', 'btap', 'path_set'
            max_row_size += 3

        return max_row_size