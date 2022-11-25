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

### User instructions

*To be written by you during the week*

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
