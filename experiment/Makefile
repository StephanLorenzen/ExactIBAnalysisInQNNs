experiments : clean link

# Link to python package
link:
	ln -s ../IB IB

data:
	./helpers/get_data.sh

# Run experiment
quantize:
	python helpers/quantize.py SYN-Tanh 8
	python helpers/quantize.py SYN-ReLU 8
MNIST-Bottleneck:
	python helpers/quantize.py MNIST-Bottleneck 8 20
binning:
	python helpers/binning.py SYN-Tanh
	python helpers/binning.py SYN-ReLU
quantize-extra:
	python helpers/quantize.py SYN-Tanh 4
	python helpers/quantize.py SYN-ReLU 4
	python helpers/quantize.py SYN-Tanh 32
	python helpers/quantize.py SYN-ReLU 32
	python helpers/quantize.py MNIST-10 8 20

# Create latex plots
plot-quantize:
	mkdir -p plots
	python helpers/plot.py quantize
	# I-planes
	cd helpers/latex; lualatex SYN_8b_ReLU_IP
	cd helpers/latex; lualatex SYN_8b_Tanh_IP
	# Accuarcy
	cd helpers/latex; lualatex SYN_accuracy
	# Clean and move
	cd helpers/latex; rm -f SYN_*.{log,aux}; rm -rf quantize
	mv helpers/latex/SYN_*.pdf plots/
plot-MNIST-Bottleneck:
	python helpers/plot.py MNIST-Bottleneck
	cd helpers/latex; lualatex REAL_8b_BN_IP_accuracy
	cd helpers/latex; lualatex REAL_8b_BN_scatter
	# Clean and move
	cd helpers/latex; rm -f REAL_8b_BN_*.{log,aux}; rm -rf MNIST-Bottleneck
	mv helpers/latex/REAL_8b_BN_*.pdf plots/
plot-binning:
	mkdir -p plots
	python helpers/plot.py binning
	cd helpers/latex; lualatex SYN_bin_ReLU_IP
	cd helpers/latex; lualatex SYN_bin_Tanh_IP
	cd helpers/latex; rm -f SYN_bin_*.{log,aux}; rm -rf binning
	mv helpers/latex/SYN_bin_*.pdf plots/
plot-quantize-extra:
	mkdir -p plots
	python helpers/plot.py quantize-extra
	cd helpers/latex; lualatex SYN_4b_IP
	cd helpers/latex; lualatex SYN_4b_accuracy
	cd helpers/latex; lualatex SYN_32b_IP
	cd helpers/latex; lualatex SYN_32b_accuracy
	cd helpers/latex; lualatex REAL_8b_10_IP_accuracy
	cd helpers/latex; rm -f SYN_*.{log,aux}; rm -f REAL_8b_10_IP_accuracy.{log,aux}; rm -rf quantize-extra
	mv helpers/latex/SYN_{4,32}b_*.pdf plots/
	mv helpers/latex/REAL_8b_10_IP_accuracy.pdf plots/

# Clean
clean:
	rm -f IB
	#rm -rf out
	rm -rf plots
