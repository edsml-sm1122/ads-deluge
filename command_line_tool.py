import argparse
from flood_tool import tool

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--postcodes', help='postcodes for prediction(str of strs)', type=str, action='store')
parser.add_argument('-oe', '--OSGB36_eastings', help='OSGB36 eastings locations for prediction(str of numbers)', type=str, action='store')
parser.add_argument('-on', '--OSGB36_northings', help='OSGB36 northings locations for prediction(str of numbers)', type=str, action='store')
parser.add_argument('-wo', '--WGS84_longitudes', help='WGS84_longitudes locations for prediction(str of numbers)', type=str, action='store')
parser.add_argument('-wa', '--WGS84_latitudes', help='WGS84_latitudes locations for prediction(str of numbers)', type=str, action='store')

parser.add_argument('-g1', '--get_flood_class_from_postcodes', help='get_flood_class_from_postcodes', action = 'store_true')
parser.add_argument('-g2', '--get_median_house_price_estimate', help='get_median_house_price_estimate_from_postcode', action = 'store_true')
parser.add_argument('-g3', '--get_local_authority_estimate_postcodes', help='get_local_authority_estimate_postcodes', action = 'store_true')
parser.add_argument('-g4', '--get_total_value', help='get_total_value', action = 'store_true')
parser.add_argument('-g5', '--get_annual_flood_risk', help='get_annual_flood_risk', action = 'store_true')

parser.add_argument('-g1_OSGB', '--get_flood_class_from_OSGB36_locations', help='get_flood_class_from_OSGB36_locations', action = 'store_true')
parser.add_argument('-g3_OSGB', '--get_local_authority_estimate_from_OSGB36_locations', help='get_local_authority_estimate_from_OSGB36_locations', action = 'store_true')
parser.add_argument('-g5_OSGB', '--get_annual_flood_risk_from_OSGB36', help='get_annual_flood_risk_from_OSGB36', action = 'store_true')

parser.add_argument('-g1_WGS', '--get_flood_class_from_WGS84_locations', help='get_flood_class_from_WGS84_locations', action = 'store_true')
parser.add_argument('-g3_WGS', '--get_local_authority_estimate_latitude_longitude', help='get_local_authority_estimate_latitude_longitude', action = 'store_true')
parser.add_argument('-g5_WGS', '--get_annual_flood_risk_from_WGS84', help='get_annual_flood_risk_from_WGS84', action = 'store_true')


args = parser.parse_args()

def g1(model, postcode):
    result = model.get_flood_class_from_postcodes(postcode)
    print('\nPredicted flood class from postcodes:')
    print(result)

def g2(model, postcode):
    result = model.get_median_house_price_estimate(postcode)
    print('\nPredicted median house price from postcodes:')
    print(result)

def g3(model, postcode):
    result = model.get_local_authority_estimate_postcodes(postcode)
    print('\nPredicted local authority from postcodes:')
    print(result)

def g4(model, postcode):
    result = model.get_total_value(postcode)
    print('\nPredicted total value from postcodes:')
    print(result)

def g5(model, postcode):
    result = model.get_annual_flood_risk(postcode)
    print('\nPredicted annual flood rirsk from postcodes:')
    print(result)

def g1_OSGB(model, OSGB36_eastings, OSGB36_northings): #python command_line_tool.py -oe "552132.0,527448.0" -on "129270.0,106738.0" -g1_OSGB
    result = model.get_flood_class_from_OSGB36_locations(OSGB36_eastings, OSGB36_northings)
    print('\nPredicted flood class from OSGB36 locations:')
    print(result)

def g3_OSGB(model, OSGB36_eastings, OSGB36_northings): #python command_line_tool.py -oe "552132.0,527448.0" -on "129270.0,106738.0" -g3_OSGB
    result = model.get_local_authority_estimate(OSGB36_eastings, OSGB36_northings)
    print('\nPredicted local authority from OSGB36 locations:')
    print(result)

def g5_OSGB(model, OSGB36_eastings, OSGB36_northings): #python command_line_tool.py -oe ",527448.0,527448.0" -on "129270.0,106738.0" -g5_OSGB
    result = model.get_annual_flood_risk_from_OSGB36(OSGB36_eastings, OSGB36_northings)
    print('\nPredicted annual flood risk from OSGB36 locations:')
    print(result)

def g1_WGS(model, WGS_longitudes, WGS_latitudes): #python command_line_tool.py -wa "51.52122296" -wo "0.18420341" -g1_WGS
    result = model.get_flood_class_from_WGS84_locations(WGS_longitudes, WGS_latitudes)
    print('\nPredicted flood class from WGS84 locations:')
    print(result)

def g3_WGS(model, WGS_latitudes, WGS_longitudes): #python command_line_tool.py -wa "51.52122296" -wo "0.18420341" -g3_WGS
    result = model.get_local_authority_estimate_latitude_longitude(WGS_latitudes, WGS_longitudes)
    print('\nPredicted local authority from WGS84 locations:')
    print(result)

def g5_WGS(model, WGS_latitudes, WGS_longitudes): #python command_line_tool.py -wa "51.52122296" -wo "0.18420341" -g5_WGS
    result = model.get_annual_flood_risk_from_WGS84(WGS_latitudes, WGS_longitudes)
    print('\nPredicted annual flood risk from WGS84 locations:')
    print(result)


if __name__=="__main__":
    if args.postcodes:
        postcodes = [str(item) for item in args.postcodes.split(',')]
    if args.OSGB36_eastings and args.OSGB36_northings:
        eastings = [float(item) for item in args.OSGB36_eastings.split(',')]
        northings = [float(item) for item in args.OSGB36_northings.split(',')]
    if args.WGS84_longitudes and args.WGS84_latitudes:
        longitudes = [float(item) for item in args.WGS84_longitudes.split(',')]
        latitudes = [float(item) for item in args.WGS84_latitudes.split(',')]

    tool_tool = tool.Tool()
    tool_tool.train()

    if args.get_flood_class_from_postcodes:
        g1(tool_tool, postcodes)
    if args.get_median_house_price_estimate:
        g2(tool_tool, postcodes)
    if args.get_local_authority_estimate_postcodes:
        g3(tool_tool, postcodes)
    if args.get_total_value:
        g4(tool_tool, postcodes)
    if args.get_annual_flood_risk:
        g5(tool_tool, postcodes)

    if args.get_flood_class_from_WGS84_locations:
        g1_WGS(tool_tool, longitudes, latitudes)
    if args.get_local_authority_estimate_latitude_longitude:
        g3_WGS(tool_tool, latitudes, longitudes)
    if args.get_annual_flood_risk_from_WGS84:
        g5_WGS(tool_tool, latitudes, longitudes)
