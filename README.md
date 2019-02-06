# IA080-classifiers-comparison
IA080 Knowledge discovery seminar


### Quick guide
```
$ cd data/extra-data/ && unzip *.zip && cd ../..
$ cd data/metal-data/ && unzip *.zip && cd ../..
$ cd data/weapon-data/ && bash download_data.sh && unzip *.zip && cd ../..
$
$ python3 -m venv venv
$ . venv/bin/activate
$ pip install -r requirements.txt
$ cd src
$ python runner.py 2>/dev/null > output/my_output.csv
```


### XCDG data generator
In the `/generator` folder you can find the XCDG.jar along with example input files used for the creation of data in `/extra-data`. The generator can be called from the command line with

`java -jar XCDG.jar input.xml output.data`

Or you can output the data in the `.csv` or `.arff` format. For further information on XCDG, please refer to the corresponding [bachelor thesis](https://is.muni.cz/auth/th/q7e3d/hetlerovic_bachelor_thesis.pdf), namely chapter 5.
