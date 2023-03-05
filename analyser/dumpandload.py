import pickle
from analyser.mask_feature import MaskFeature
from analyser.celltracer import CellTracer


def dump_maskfeature(maskobj: MaskFeature, path=None):
    output = {}
    output["data"] = maskobj.__array__()
    output["instance_properties"] = maskobj.instance_properties
    output["cost"] = maskobj.cost()
    if path:
        with open(path, 'wb') as f:
            pickle.dump(output, f)
    else:
        return output


def load_maskfeature(inputs=None, path=None):
    if inputs is None:
        if path is not None:
            with open(path, 'rb') as f:
                inputs = pickle.load(f)
        else:
            return None
    maskobj = MaskFeature(inputs["data"])
    maskobj.instance_properties = inputs["instance_properties"]
    maskobj.set_cost(inputs["cost"])
    return maskobj


def dump_celltracer(ter: CellTracer, path=None):
    output = {}
    output["mask"] = ter.__array__()
    output["frame_number"] = ter.frame_number,
    output["obj_number"] = ter.obj_number,
    output["maskobj"] = {}
    for k in ter.maskobj:
        output["maskobj"][k] = dump_maskfeature(ter.maskobj[k])
    output["traced_image"] = ter.traced_image,
    output["obj_property"] = ter.obj_property
    output["trace_calendar"] = ter.trace_calendar,
    output["distance"] = ter.distance,
    output["props"] = ter.props
    output["coords"] = ter.coords
    if path:
        with open(path, 'wb') as f:
            pickle.dump(output, f)
    else:
        return output


def load_celltracer(path):
    with open(path, 'rb') as f:
        inputs = pickle.load(f)
    mask = inputs["mask"]
    ter = CellTracer(mask)
    ter.frame_number = inputs["frame_number"][0]
    ter.obj_number = inputs["obj_number"][0]
    ter.maskobj = {}
    for k in range(0, mask.shape[0]):
        ter.maskobj[k] = load_maskfeature(inputs["maskobj"][k])
    ter.traced_image = inputs["traced_image"][0]
    ter.obj_property = inputs["obj_property"]
    ter.trace_calendar = inputs["trace_calendar"][0]
    ter.distance = inputs["distance"][0]
    ter.props = inputs["props"]
    ter.coords = inputs["coords"]
    return ter
