#!/bin/bash
#Assuming this is launched from the source/ directory

mkdir -p ../datasets/hetrec
wget http://files.grouplens.org/datasets/hetrec2011/hetrec2011-movielens-2k-v2.zip
mv hetrec2011-movielens-2k-v2.zip ../datasets/hetrec/
cd ../datasets/hetrec/
unzip hetrec2011-movielens-2k-v2.zip
rm hetrec2011-movielens-2k-v2.zip

