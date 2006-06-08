#!/bin/sh


echo "# Creating tmp directory"
cd ..
mkdir tmp
cp -r universvm tmp
cd tmp
echo "# Removing unnecessary files"
rm -rf universvm/.svn
rm -rf universvm/svqp2/.svn
rm universvm/universvm*
rm universvm/libsvm2bin
rm universvm/bin2libsvm
echo "# Packing source code"
tar -czf source_v1.tar.gz universvm

echo "# Copying it to the server"
cp source_v1.tar.gz /net/homepage/export/www/campus/kyb/bs/people/fabee/

echo "# Cleaning up"
cd ..
rm -rf tmp

echo "# Done"
