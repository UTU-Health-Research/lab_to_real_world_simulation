import yaml
import json



class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)   
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def load_yaml(yaml_file):
    #yaml_filepath = os.path.join(os.getcwd(), yaml_file)
    with open(yaml_file, 'r') as file:
        args = yaml.safe_load(file)       
    
    if args['verbose']:
        print('args: {}'.format(args))
    
    return dict2obj(args)




