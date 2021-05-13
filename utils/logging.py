def extract_tfm_string(tfm):
    info = tfm.get_dict_with_id()
    name = info["__class_fullname__"].split(".")[-1]
    prob = info["p"]
    return f"{name}__p-{prob}"
