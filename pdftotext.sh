#!/bin/bash


for file in $(find pdf_files/prx* -maxdepth 4)
do
    if [ ! -d "${file}" ] ; then
	echo "Processing $file file..."
	var=$file
	replace="txt"
	replace1="pdf_texts"
	tmp=${var//pdf/$replace}
	tmp1=${tmp//txt_files/$replace1}
	nf="$tmp1"
	echo $nf
	A1="$(cut -d'/' -f1 <<< $nf)"
	A2="$(cut -d'/' -f2 <<< $nf)"
	A3="$(cut -d'/' -f3 <<< $nf)"
	A4="$(cut -d'/' -f4 <<< $nf)"
	A5="$(cut -d'/' -f5 <<< $nf)"
	echo "$A1/$A2/$A3/$A4/$A5"
	mkdir -p "$A1/$A2/$A3/$A4/"
	pdftotext "$file" "$A1/$A2/$A3/$A4/$A5"
    else
	echo "Entering: ${file}"
    fi
done






