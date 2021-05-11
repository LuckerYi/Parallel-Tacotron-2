set -euxo pipefail

data_dir="dataset"
pushd `pwd`
mkdir -p $data_dir
cd $data_dir
wget -q --show-progress https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xjf LJSpeech-1.1.tar.bz2

# extract transcript
cd LJSpeech-1.1
cut -f 3 -d '|' ./metadata.csv | \
sed 's/Mr\./Mister/g' | \
sed 's/Mrs\./Misess/g' | \
sed 's/Dr\./Doctor/g' | \
sed 's/No\./Number/g' | \
sed 's/St\./Saint/g' | \
sed 's/Co\./Company/g' | \
sed 's/Jr\./Junior/g' | \
sed 's/Maj\./Major/g' | \
sed 's/Gen\./General/g' | \
sed 's/Drs\./Doctors/g' | \
sed 's/Rev\./Reverend/g' | \
sed 's/Lt\./Lieutenant/g' | \
sed 's/Hon\./Honorable/g' | \
sed 's/Sgt\./Sergeant/g' | \
sed 's/Capt\./Captain/g' | \
sed 's/Esq\./Esquire/g' | \
sed 's/Ltd\./Limited/g' | \
sed 's/Col\./Colonel/g' | \
sed 's/Ft\./Fort/g'  > transcript.txt

# extract filenames
cut -f 1 -d '|' ./metadata.csv > filenames.txt

# convert text to phonemes
mkdir phoneme_temp
split -l 200 transcript.txt phoneme_temp/transcript_
for filename in phoneme_temp/transcript_*; do
  echo $filename
  phonemize -l en-us -b espeak -j 4 --strip --with-stress --preserve-punctuation $filename -o "$filename.phon"
done
cat phoneme_temp/transcript_*.phon > phonemes.txt
rm -r phoneme_temp

popd