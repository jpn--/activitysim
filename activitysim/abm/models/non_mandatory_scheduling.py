# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import pandas as pd

from activitysim.core import simulate as asim
from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import pipeline
from activitysim.core import timetable as tt

from .util import expressions
from .util.vectorize_tour_scheduling import vectorize_tour_scheduling
from activitysim.core.util import assign_in_place


logger = logging.getLogger(__name__)
DUMP = False


@inject.injectable()
def tour_scheduling_nonmandatory_spec(configs_dir):
    return asim.read_model_spec(configs_dir, 'tour_scheduling_nonmandatory.csv')


@inject.injectable()
def non_mandatory_tour_scheduling_settings(configs_dir):
    return config.read_model_settings(configs_dir, 'non_mandatory_tour_scheduling.yaml')


@inject.step()
def non_mandatory_tour_scheduling(tours,
                                  persons_merged,
                                  tdd_alts,
                                  tour_scheduling_nonmandatory_spec,
                                  non_mandatory_tour_scheduling_settings,
                                  chunk_size,
                                  trace_hh_id):
    """
    This model predicts the departure time and duration of each activity for non-mandatory tours
    """

    trace_label = 'non_mandatory_tour_scheduling'

    tours = tours.to_frame()
    persons_merged = persons_merged.to_frame()

    non_mandatory_tours = tours[tours.tour_category == 'non_mandatory']

    logger.info("Running non_mandatory_tour_scheduling with %d tours" % len(tours))

    constants = config.get_model_constants(non_mandatory_tour_scheduling_settings)

    # - run preprocessor to annotate choosers
    preprocessor_settings = \
        non_mandatory_tour_scheduling_settings.get('preprocessor_settings', None)
    if preprocessor_settings:

        locals_d = {}
        if constants is not None:
            locals_d.update(constants)

        expressions.assign_columns(
            df=non_mandatory_tours,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label)

    tdd_choices = vectorize_tour_scheduling(
        non_mandatory_tours, persons_merged,
        tdd_alts, tour_scheduling_nonmandatory_spec,
        constants=constants,
        chunk_size=chunk_size,
        trace_label=trace_label)

    assign_in_place(tours, tdd_choices)
    pipeline.replace_table("tours", tours)

    # updated df for tracing
    non_mandatory_tours = tours[tours.tour_category == 'non_mandatory']

    tracing.dump_df(DUMP,
                    tt.tour_map(persons_merged, non_mandatory_tours, tdd_alts),
                    trace_label, 'tour_map')

    if trace_hh_id:
        tracing.trace_df(non_mandatory_tours,
                         label="non_mandatory_tour_scheduling",
                         slicer='person_id',
                         index_label='tour_id',
                         columns=None,
                         warn_if_empty=True)