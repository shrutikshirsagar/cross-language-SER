#!/bin/bash

#### In this pipleine, Step 1: we first get the enhancement audio files from the pretrained models from the speechbrain
#### Step 2: we get the s[eech transcripts and BERT feature based on the transcripts
#### step 3: we extract modulation feature and we convert enhancemnet audio to 16 bit in order to extract egemap fetaures
#### step 4: Finally, we run neural network weher we first trained the model and saved it so that we can use the same trained model for tetsing different noise types and noise leveles.

conda activate tensorflow_project1

cd //media/amrgaballah/Backup_Plus/exp_J3/cross-language/F_train_G_test/codes/
python BiLSTM_feat_SA.py

cd //media/amrgaballah/Backup_Plus/exp_J3/cross-language/F_train_G_test/codes/
python BiLSTM_feat_BOW_SA.py

cd //media/amrgaballah/Backup_Plus/exp_J3/cross-language/F_train_G_test/codes/
python BiLSTM_feat_SA_BOW.py
