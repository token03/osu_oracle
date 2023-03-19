# osu!oracle (WIP)

osu!oracle is a machine learning model that categorizes osu! beatmaps into existing categories such as aim, streams, speed, tech, etc. The primary goal is to enable swift and accurate categorization of beatmaps without the need to access the editor or play them manually.

## Overview
### [Link to Google Drive](https://drive.google.com/file/d/1o5fbixI9xK_WF_GFhvNOI77D0HW-KeZO/view?usp=share_link) | [Link to Google Colab](https://colab.research.google.com/drive/1vVEpzWpSfArfHxL41sSdiXFtE-0U22HN?usp=sharing) 


> Model is currently trained on collections from [osu!collector](https://osucollector.com/) where the beatmaps are mostly from tournament pools and range from `~ 5.3*` to `~ 8.5*` range.

> Current categories are `aim, alt, tech, and streams` with hopefully more to come



## Requirements
```
- Python 3.6 or higher
- tensorflow
- keras
- numpy
- scikit-learn
```
## Getting Started

1. Download the model from the Google Drive link above
2. Open the directory in a terminal 
	- Make sure you have all the requirements installed. If you don't, run the following command: `pip install <w/e>`
3. Run the following script to test the current iteration of the model:
```
python test_model.py <beatmap_id>
```
> ex. to test Blue Zenith's top diff, take the last sequence of digits (the beatmap_id) https://osu.ppy.sh/beatmapsets/292301#osu/658127 and run the following command:
```
python test_model.py 658127
```

4. Output should look something like this:
![Image of output](data\example.png)

5. You could also try the Google Colab but it's still a WIP.

## Known Issues
- Low SR (< ~5) maps are not being categorized properly at all 
	- ex. 1264763
- Maps with multiple skillsets sometimes get categorized weirdly 
	- ex. Marianne (644971) as tech/alt
- Gamemodes other than standard are currently not supported and will probably break
- Extremely short and extremely long maps are sometimes categorized weirdly




## Disclaimer 

This project is still a **massive WIP**. The model is in a functioning state but still needs tweaking along with a host of other issues. This is my first time working with both Python and ML so I'm sure there are better ways to do things. 

