import analyser.common as common

DIVISION = 1000
MIN_HITS = 2

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

CELL_IMAGE_PROPERTY = [
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
