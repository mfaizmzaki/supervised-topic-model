# MSc-Thesis
Supervised Topics Over Time model

This repository contains all the source codes used in UCL MSc Web Science and Big Data Analytics project : Discovering patterns of UN  peacekeeping activies in Ivory Coast using supervised topic modelling. An important note is that the source codes in this repository only deals with the modelling process. It does not include the data collection source codes. The codes will be released when there is mutual agreement between me and Ms. Hannah Smidt as my co-supervisor. However, the original data can be accessed at UN peacekeeping mission in Ivory Coast website: http://www.onuci.org/

# Files in this repo
1. stot_model.py
2. main_stot.py
3. timing.py
4. coherence.py
5. visualize_topics.py
6. stopwords.txt
7. locations.txt
8. seedwords.txt

# Running the codes
1. Ensure that the dataset is prepared. Each article from the website is in the form of a text file with the following format:
              
            Title:
            Date:
            Text:
            Locations:

2. Edit the paths in main_stot.py accordingly
3. The number of iterations and parameters of the model are set in stot_model.py in line 189 (method initParam). Change if necessary. 
4. Run main_stot.py
5. The model will be stored in .pickle format at the path set in main_stot.py
6. Edit the paths in visualize_topics.py and coherence.py accordingly.
7. Run visualize_topics.py to view the results.
8. Run coherence.py to view the coherence score.
