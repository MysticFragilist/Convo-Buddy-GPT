import configparser

config = configparser.ConfigParser()
config.read('config.cfg')
config.sections()

def getConfig(key):
    return config['DEFAULT'][key]