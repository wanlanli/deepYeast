# Cell Image Properties
# CELL_IMAGE_PROPERTY
import analyser.common as common

DIVISION = 1000
MIN_HITS = 2
TIP_ANGLE = 15

REGION_TABLE_VALUE = [
     common.IMAGE_LABEL,
     common.IMAGE_CENTER,
     common.IMAGE_ORIENTATION,
     common.IMAGE_MAJOR_AXIS,
     common.IMAGE_MINOR_AXIS,
     common.IMAGE_AREA,
     common.IMAGE_BOUNDING_BOX,
     common.IMAGE_ECCENTRICITY
     ]

TRACE_IMAGE_PROPERTY = [
     common.IMAGE_LABEL,
     common.IMAGE_CENTER_LIST[0],
     common.IMAGE_CENTER_LIST[1],
     common.IMAGE_ORIENTATION,
     common.IMAGE_MAJOR_AXIS,
     common.IMAGE_MINOR_AXIS,
     common.IMAGE_AREA,
     common.IMAGE_BOUNDING_BOX_LIST[0],
     common.IMAGE_BOUNDING_BOX_LIST[1],
     common.IMAGE_BOUNDING_BOX_LIST[2],
     common.IMAGE_BOUNDING_BOX_LIST[3],
     common.IMAGE_ECCENTRICITY,
     common.IMAGE_SEMANTIC_LABEL,
     common.IMAGE_INSTANCE_LABEL,
     common.IMAGE_IS_BORDER,
     common.IMAGE_COORDINATE,
     ]

CELL_IMAGE_PROPERTY = TRACE_IMAGE_PROPERTY

OBJ_DISTANCE_COLUMNS = [
    common.CENTER_DISTANCE,
    common.NEARNEST_DISTANCE,
    common.NEARNEST_POINT_INDEX[0],
    common.NEARNEST_POINT_INDEX[1],
]

TRACKE_PROPERTY = [
     common.OBJ_ID,
     common.OBJ_START,
     common.OBJ_END,
     common.OBJ_LIFE_SPAN,
]

CELL_TRACKE_PROPERTY = [
     common.CELL_ID,
     common.CELL_GENERATION,
     common.CELL_START,
     common.CELL_END,
     common.CELL_MOTHER,
     common.CELL_FATHER,
     common.CELL_DIVISION_FLAGE,
     common.CELL_DAUGHTER[0],
     common.CELL_DAUGHTER[1],
     common.CELL_FUSION_FLAGE,
     common.CELL_SPOUSE,
     common.CELL_SON,
     common.CELL_LIFE_SPAN,
]


IMAGE_LABEL = "label"
IMAGE_CENTER = "centroid"
IMAGE_CENTER_LIST = ['centroid_0', 'centroid_1']
IMAGE_ORIENTATION = "orientation"
IMAGE_MAJOR_AXIS = "axis_major_length"
IMAGE_MINOR_AXIS = "axis_minor_length"
IMAGE_AREA = "area"
IMAGE_BOUNDING_BOX = "bbox"
IMAGE_BOUNDING_BOX_LIST = ['bbox_0', 'bbox_1', 'bbox_2', 'bbox_3']
IMAGE_ECCENTRICITY = "eccentricity"
IMAGE_COORDINATE = "coords"
IMAGE_SEMANTIC_LABEL = "semantic"
IMAGE_INSTANCE_LABEL = "instance"
IMAGE_IS_BORDER = "is_out_of_border"

IMAGE_CONTOURS_LENGTH = 60

# TRACE_FEATURE_NAME
CELL_ID = OBJ_ID = "identity"
CELL_START = OBJ_START = "start_time"
CELL_END = OBJ_END = "end_time"
CELL_LIFE_SPAN = OBJ_LIFE_SPAN = "life_time"
CELL_TABEL_ARG = OBJ_TABEL_ARG = 'arg'


CELL_GENERATION = "generation"
CELL_MOTHER = "mother"
CELL_FATHER = "father"
CELL_DIVISION_FLAGE = "is_divided"
CELL_DAUGHTER = ["daughter_1", "daughter_2"]
CELL_FUSION_FLAGE = "is_fusioned"
CELL_SPOUSE = "spouse"
CELL_SON = "son"

# OVERLAP BATCH
OVERLAP_VMIN = 0.1
OVERLAP_VMAX = 0.75
DIVISION_LABEL = 3
FUSION_LABEL = 2

# FLOURESCENT_INFO
NUM_FLOURESCENT = "num_flourescent_channel"
CHANNEL_PREDICTION = "channel_prediction"

# TWO regions' features
CENTER_DISTANCE = "center_dist"
NEARNEST_DISTANCE = "nearnest_dis"
NEARNEST_POINT_INDEX = ["nearnest_point_x_index", "nearnest_point_y_index"]
ANGLE_POINT_CENTER = ["angle_x", "angle_y"]
TIME_GAP = "timegap"
#


DISTANCE_COLUMNS = ['index_x', 'index_y', 'center_dist', 'nearnest_dis', 'nearnest_point_x_index', 'nearnest_point_y_index']

REGION_NAMES = ['label','centroid_0','centroid_1','orientation',
'axis_major_length','axis_minor_length',
            'area','bbox_0', 'bbox_1', 'bbox_2', 'bbox_3', 'eccentricity', 
            'coords', 'semantic', 'instance', 'out_of_border']

PROP_NAMES = ['label','centroid_0','centroid_1','orientation','axis_major_length','axis_minor_length',
            'area','bbox_0', 'bbox_1', 'bbox_2', 'bbox_3', 'eccentricity', 'semantic', 'instance']

CELL_PROP_COLUMNS = ['centroid_0','centroid_1','orientation','axis_major_length','axis_minor_length',
            'area','bbox_0', 'bbox_1', 'bbox_2', 'bbox_3', 'eccentricity', 'coords']

TRACING_FEATRUE_NAMES = ["indentity","generation", "mother", "father", 
                         "start_time", "end_time", #0-5
                        "is_divided", "sub_1", "sub_2", "is_fusioned", "spouse","son",#6-11
                   "status",#12
                   "instance","semantic",#13-14
                   "ch1_pos","ch2_pos",#15-16,
                   "life_time",#17
                   "arg",#18
                  ]
