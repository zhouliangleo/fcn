

# Parameters!
mainFolder="net_runs"
net="fcn_shallow"
subFolder=${netType}_${net}
##################################################################### 
mkdir -p ${mainFolder}/${subFolder}

echo "Current network folder: "
echo ${mainFolder}/${subFolder}



python3 -u train.py --tfdir ${mainFolder}/${subFolder} 2>&1 | tee -a ${mainFolder}/${subFolder}/log.txt    




