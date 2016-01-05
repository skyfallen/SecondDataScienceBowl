#!/bin/bash
# Convert all .dcm files to .bmp in the indicated
# directory. All converted files will be located in /bmp 
# subdirectory
#
# Usage: ./dcm2bmp.sh PATH
# Example: ./dcm2bmp.sh /storage/hpc_dmytro/Kaggle/SDSB/images/bmpsamples
#
SOURCE=$1
SINK="${SOURCE}bmp/"
echo $SINK

for dcm_name in $(find $SOURCE -name '*.dcm')
    do
        # Parse new file namees from old ones
	bmp_name=`echo ${dcm_name} | sed 's/dcm$/bmp/' | sed 's:.*/::'`
	bmp_name="$SINK$bmp_name"
        echo convert ${dcm_name} ${bmp_name}
	
	# Check if folder $bmp_name exists, if not create it
	if [ ! -d $SINK ]; then
  		mkdir $SINK
	fi	

	# Use Imagemagick
        convert ${dcm_name} ${bmp_name}
        retval=$?
    if [[ $retval != 0 ]]
        then
            echo "Error converting ${dcm_name} (retval=${retval})"
        exit 1
    fi
done
exit 0
