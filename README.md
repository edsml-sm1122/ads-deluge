# Flood Risk Prediction tool

## Deadlines
-  *code 12pm GMT Friday 25th November 2022*
-  *presentation/ one page report 4pm GMT Friday 25th November 2022*

### Key Requirements

Your project must provide the following:

 - at least one analysis method to estimate a number of properties for unlabelled postcodes extrapolated from sample data which is provided to you:
    - Flood risk (on a 10 point scale).
    - Median house price.
 - at least one analysis method to estimate the Local Authority & flood risk of arbitrary locations. 
 - a method to find the rainfall and water level near a given postcode from provided rainfall, river and tide level data, or by looking online.

 You should also provide visualization and analysis tools for the postcode, rainfall, river & tide data provided to you, ideally in a way which will identify potential areas at immediate risk of flooding.
 
 Your code should have installation instructions and basic documentation, either as docstrings for functions & class methods, a full manual or both.

![London postcode density](images/LondonPostcodeDensity.png)
![England Flood Risk](images/EnglandFloodRisk.png)
![UK soil types](images/UKSoilTypes.png)

This README file *should be updated* over the course of your group's work to represent the scope and abilities of your project.

### Assessment

 - your code will be assessed for its speed (both at training and prediction) & accuracy.
 - Your code should include tests of its functionality.
 - Additional marks will be awarded for high code quality and a clean, well organised repository.

 ### Installation Guide

**Prerequisite**

This project uses conda as a package manager. You should have conda configured on your local machine before installing the project. 

```bash
conda -V
``` 

**Installation and configuration**

* To install the project, first clone the repository: 

```bash
git clone https://github.com/ese-msc-2022/ads-deluge-Thames.git
```

* Go to the git repository on you local computer: 

```bash
cd ads-deluge-Thames
```

* Then configure the conda environment:

```bash
conda env create -f environment.yml
```

* Activate the `deluge` conda environment:

```bash
conda activate deluge
```

* Once you have finished using the tool, you can deactivate the conda environment using the following command:

```bash
conda deactivate
```

### Who is this software for?

-	Property agents to calculate risk of their properties
-	Government to prepare for flooding
- Environment NGOs
- Insurance companies

### User instructions

#### Risk Tool
The tool.py file combines the main functionality of the flood risk tool. 
<br>
The first step to use the tool is to import tool from flood_tool and initialise the class: 
* import tool from flood_tool
* tool = tool.Tool(UNLABELLED, LABELLED)

Then train all models that can be used by giving a labelled set of samples:
* tool.train()

See below for a description of the main functionality in tool.py and an example on how to use it. 

1. Convert between UK ordanance survey easting/northing coordinates, GPS latitude & longitude, and postcode.
   - To find eastings and northings from postcode:
      - tool.get_easting_northing(postcodes=['BN1 5PF'])
   - To find latitude and longitude from postcode:
      - tool.get_lat_long(postcodes=['BN1 5PF'])
   - To find postcode from easting and northing:
      - tool.get_postcode_from_OSGB36(eastings=[417997.0], northings=[97342.0])
   - To find postcode from latitude and longitude:
      - tool.get_postcodes_from_WGS84(latitudes=[50], longitudes=[0])

2. Predict the Local Authority for arbitrary locations (use tool.get_local_authority_methods() to see available methods).
   - tool.get_local_authority_estimate(eastings=[417997.0, 535049.0], northings=[97342.0, 169939.0], method=1)

3. Predict the median house price for input postcodes (use tool.get_house_price_methods() to see available methods).
   - tool.get_median_house_price_estimate(postcodes=['BN1 5PF'], method=1)

4. Predict flood probability for input postcodes or arbitrary locations (use tool.get_flood_class_from_locations_methods() to see available methods)
   - tool.get_flood_class_from_postcodes(postcodes=['BN1 5PF'], method=1)
   - tool.get_flood_class_from_OSGB36_locations(eastings=[417997.0, 535049.0], northings=[97342.0, 169939.0], method=1)
   - tool.get_flood_class_from_WGS84_locations(longitudes=[0], latitudes=[50], method=1)

5. Predict flood risk for input postcodes or arbitrary locations.
   - tool.get_annual_flood_risk(postcodes['BN1 5PF'])
   - tool.get_annual_flood_risk_from_WGS84(longitudes=[0], latitudes=[50])
   - tool.get_annual_flood_risk_from_OSGB36(eastings=[417997.0, 535049.0], northings=[97342.0, 169939.0])

#### Data Visualiser
1. In the command line run 'python DataVisualization.py'. This may take a few minutes to run.
2. Open file 'a_map.html' and jump to the online interactive map.
3. Click on the buttons to visualise different types of data.
4. Open the DataVisualization.py file and change the file path passed to the variable 'unlabeled' to plot predictions for a different dataset. Go back to step 1 to run the program. 

![visualiser](images/visualiser_screenshot.png)

## Usage of command line:

|input data|usage|
|---|---|
| postcodes | -p |
|OSGB36_eastings|-oe|
|OSGB36_northings|-on|
|WGS84_longitudes|-wo|
|WGS84_latitudes|-wa|

- The expected input format will be a string, and each element is seperated by a comma. 
- For example: **"CT2 8AA,TN28 8XN"**

|function|usage|
|---|---|
|get_flood_class_from_postcodes| -g1 |
|get_median_house_price_estimate| -g2 |
|get_local_authority_estimate_postcodes| -g3 |
|get_total_value | -g4 |
|get_annual_flood_risk | -g5 |
|get_flood_class_from_OSGB36_locations | -g1_OSGB |
|get_local_authority_estimate_from_OSGB36_locations | -g3_OSGB |
|get_annual_flood_risk_from_OSGB36 | -g5_OSGB |
|get_flood_class_from_WGS84_locations| -g1_WGS |
|get_local_authority_estimate_latitude_longitude| -g3_WGS |
|get_annual_flood_risk_from_WGS84 | -g5_WGS | 

**Example** `(get annual flood risk from sets of postcodes)` :

\>>> python command_line_tool.py -p "CT2 8AA,TN28 8XN" -g5

\>>> Predicted annual flood rirsk from postcodes:

CT2 8AA     5723.422968

TN28 8XN      18.231949

dtype: float64


**Example** `(get local authority estimate from OSGB36 locations)` :

\>>> python command_line_tool.py -oe "552132.0,527448.0" -on "129270.0,106738.0" -g3_OSGB

\>>> Predicted flood class from OSGB36 locations:

552132.0  129270.0              Wealden

527448.0  106738.0    Brighton and Hove

Name: localAuthority, dtype: object

### How does this software works ?

This software is composed of two main part. 

* A risk tool, allowing the user to predict different properties such as the local authority, the median house price, the flood probability and the overall flood risk using different location inputs. The possible inputs are UK postcodes, easting and northing coordinates and gps coordinates. 
* A visualization tool that allows the user to have a general overview of the properties mentioned above. This tool is organized in layers so that the user can select only the information needed and can display them of different types of maps. It can also detect areas with abnormal values of rain or tides. These areas are evaluated in the `indicate_area_at_risk.ipynb`.


### A brief technical overview of the risk tool.

The main functionalities of the Risk Tool are gathered in python files in the flood_tool folder and more specifically in the tool.py file. 

The different models to predict the house median price, the local authority and the flood probability are defined as modules in the flood_tool package. Some models can be trained using different methods. 

The tool.py file contains a train function which will train the models in every methods available. It also contains the different functions that predict the wanted properties with different type of location input. 

* The local authority model, defined in the `local_authority.py`, uses a KNN classifier to predict the local authority from easting and northing
* The median price model, defined in the `median_price.py`, uses a KNN Regressor to predict the house median price from a postcode
* The flood probability model, defined in the `flood_prob.py` can be trained either by using a Random Forest Regressor or a KNN Regressor from a postcode. 

The `geo.py` file contains functions to convert easting/northing coordinates to latitude/longitudes. 

The `live.py` file contains functions to get rainfall or tidal data from a given dataset or to retrieve similar data from an API. The API called is https://environment.data.gov.uk/flood-monitoring/id/stations/. 

Finally, the flood_tool package contains a series of unit tests defined in the tests folder.



### Documentation

_This section should be updated during the week._

The code includes [Sphinx](https://www.sphinx-doc.org) documentation. On systems with Sphinx installed, this can be build by running

```
python -m sphinx docs html
```

then viewing the generated `index.html` file in the `html` directory in your browser.

For systems with [LaTeX](https://www.latex-project.org/get/) installed, a manual pdf can be generated by running

```bash
python -m sphinx  -b latex docs latex
```

Then following the instructions to process the `FloodTool.tex` file in the `latex` directory in your browser.

### Testing

The tool includes several tests, which you can use to check its operation on your system. With [pytest](https://doc.pytest.org/en/latest) installed, these can be run with

```bash
python -m pytest flood_tool
```

### Reading list

 - (A guide to coordinate systems in Great Britain)[https://webarchive.nationalarchives.gov.uk/20081023180830/http://www.ordnancesurvey.co.uk/oswebsite/gps/information/coordinatesystemsinfo/guidecontents/index.html]

 - (Information on postcode validity)[https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/283357/ILRSpecification2013_14Appendix_C_Dec2012_v1.pdf]
