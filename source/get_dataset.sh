#!/bin/bash
#Assuming this is launched from the source/ directory

mkdir -p ../datasets/hetrec
cd ../datasets/hetrec/
wget http://files.grouplens.org/datasets/hetrec2011/hetrec2011-movielens-2k-v2.zip
unzip hetrec2011-movielens-2k-v2.zip
rm hetrec2011-movielens-2k-v2.zip

#convert everything to UTF-8
ls -1 *.dat | parallel iconv -f ISO-8859-15 -t UTF-8 -o '{}' '{}'

printf "%s\n" "manually download https://1drv.ms/u/s\!AtpT6fwcpsVRhLwvS0bpCrbW03v-1w and unzip to datasets/Enriched_Netflix_Dataset/"
