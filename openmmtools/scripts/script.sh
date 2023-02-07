# ls | grep sdf | while read f; do obabel -isdf $f -omol2 -O "${f%%.*}.mol2"; done


ls | grep mol2 | while read f
do 
	mkdir ${f%%.*}_processing
	cd ${f%%.*}_processing
	antechamber -fi mol2 -i ../$f -c bcc -fo mol2 -o $f
	cat $f | grep "1 UNL" | awk '{ print $6} ' > atomtypes.txt 
	cd ..
done


