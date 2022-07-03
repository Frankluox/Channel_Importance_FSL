from importlib import import_module

def get_backbone(architecture_name, *args, **kwargs):
    architecture_module = import_module("." + architecture_name, package="architectures.backbone")
    create_model = getattr(architecture_module, "create_model")
    return create_model(*args, **kwargs)

def get_classifier(architecture_name,**kwargs):
    architecture_module = import_module("." + architecture_name, package="architectures.classifier")
    create_model = getattr(architecture_module, "create_model")
    return create_model(**kwargs)