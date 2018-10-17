#!/usr/bin/python
from osgeo import gdal
from rios import applier
from rios import fileinfo
import multiprocessing
import sys
import numpy as np



def stats(filename, nodata, i):

	gtif = gdal.Open(filename)
	data = gtif.GetRasterBand(i).ReadAsArray(0,0, gtif.RasterXSize, gtif.RasterYSize)
	metadado = {
		'min': np.min(data),
		'max': np.max(data),
		'mean': np.mean(data),
		'median': np.median(data),
		'stddev': np.std(data),
		'nodata': nodata
	}
	print(filename, metadado)
	return 

def normalize(block, stats, outputNodata):
	
	validPixels = (block != stats['nodata'])
	block[validPixels] = (block[validPixels] - stats['median']) / stats['stddev']
	block[(block == stats['nodata'])] = outputNodata

	return block

def filter(info, inputs, outputs, otherargs):
	
	print("Processing status " + str(info.getPercent()) + "%")

	norm_blue1 = normalize(inputs.image1[0:1,:,:].astype('Float32'), otherargs.stats_blue1, otherargs.output_nodata)
	norm_green1 = normalize(inputs.image1[1:2,:,:].astype('Float32'), otherargs.stats_green1, otherargs.output_nodata)
	norm_red1 = normalize(inputs.image1[2:3,:,:].astype('Float32'), otherargs.stats_red1, otherargs.output_nodata)
	norm_nir1 = normalize(inputs.image1[3:4,:,:].astype('Float32'), otherargs.stats_nir1, otherargs.output_nodata)

	norm_blue2 = normalize(inputs.image2[0:1,:,:].astype('Float32'), otherargs.stats_blue2, otherargs.output_nodata)
	norm_green2 = normalize(inputs.image2[1:2,:,:].astype('Float32'), otherargs.stats_green2, otherargs.output_nodata)
	norm_red2 = normalize(inputs.image2[2:3,:,:].astype('Float32'), otherargs.stats_red2, otherargs.output_nodata)
	norm_nir2 = normalize(inputs.image2[3:4,:,:].astype('Float32'), otherargs.stats_nir2, otherargs.output_nodata)

	outputBands = [
		norm_blue1, norm_green1, norm_red1, norm_nir1,
		norm_blue2, norm_green2, norm_red2, norm_nir2
	]

	if otherargs.hasReferenceImage:
		outputBands.append(inputs.referenceImage.astype('Float32'))
	
	outputs.normalized_stacked = np.concatenate(outputBands);

image1 = sys.argv[1]
image2 = sys.argv[2]

print("imagem 1", image1)
print("imagem 2", image2)
hasReferenceImage = (len(sys.argv) == 4)

if(hasReferenceImage):
	referenceImage = sys.argv[3]
	outputImage = sys.argv[4]
else:
	outputImage = sys.argv[3]

infiles = applier.FilenameAssociations()
outfiles = applier.FilenameAssociations()
otherargs = applier.OtherInputs()

infiles.image1 = image1
infiles.image2 = image2

outfiles.normalized_stacked = outputImage
otherargs.output_nodata = -10

otherargs.stats_blue1 = stats(infiles.image1, 0, 1)
otherargs.stats_green1 = stats(infiles.image1, 0, 2)
otherargs.stats_red1 = stats(infiles.image1, 0, 3)
otherargs.stats_nir1 = stats(infiles.image1, 0, 4)
print("estatisica da image 1 pronta")
otherargs.stats_blue2 = stats(infiles.image2, 0, 1)
otherargs.stats_green2 = stats(infiles.image2, 0, 2)
otherargs.stats_red2 = stats(infiles.image2, 0, 3)
otherargs.stats_nir2 = stats(infiles.image2, 0, 4)
print("estatisica da image 1 pronta")
otherargs.hasReferenceImage = hasReferenceImage

numCPU = multiprocessing.cpu_count()
print("CPUs rodando", numCPU)
controls = applier.ApplierControls()
controls.setNumThreads(numCPU);

if(hasReferenceImage):
	infiles.referenceImage = referenceImage
	controls.referenceImage = infiles.referenceImage

controls.setJobManagerType('multiprocessing')

applier.apply(filter, infiles, outfiles, otherargs, controls=controls)
print("processo encerrado!")