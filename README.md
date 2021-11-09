CODE FOR THE PAPER
=========================================================================================
Yueyu Hu, Shuai Yang, Wenhan Yang, Ling-Yu Duan, and Jiaying Liu. 
"Towards Coding for Human and Machine Vision: A Scalable Image Coding Approach", 
IEEE International Conference on Multimedia & Expo (ICME), London, UK, July 2020.
=========================================================================================

1_structured_edge_detection
	P. Dollar and C. L. Zitnick. Structured forests for fast edge detection. ICCV, 2013. 
	https://github.com/pdollar/edges
	get_edge.m 
		run structured edge detection over images in data/imgs/
		resulting in BMP edge images in data/edges/
	
2_autotrace
	M.Weber. AutoTrace: a program for converting bitmap to vector graphic. 1998. 
	http://autotrace.sourceforge.net/
	run_single_image.bat & run_batch_images.bat
		run autotrace over BMP edge images in data/edges/
		resulting in SVG vectorized edges in data/edges/
	
		
3_compression
	get_bsNsvg_v2.py
		The program scan the provided folders to produce the bit-streams. 
		For each image and the corresponding SVG file produced by the AutoTrace tool, 
		the program generates the quantized vectorized representation (TXT), the enhancement bitstream consisting of the RGB color samples (RGB) in data/features/
		and re-renders edges/color samples and masks (PNG) in data/decoder_img_input/
	quantize.py
		Quantize the SVG parameters and make the compact representation.
	make_ppm_parallel.py
		In the parallel way, losslessly compress the compact text representation with the PPM algorithm provided by 7zip tools.

4_decoder
	P. Isola et al. Image-to-Image Translation with Conditional Adversarial Networks. CVPR, 2017. 
	https://github.com/phillipi/pix2pix
	HV_train.py
		train network for human vision tasks		
	MV_train.py
		train network for machine vision tasks	
	HV_test.py
		test to recover the image for human vision tasks	
	MV_test.py
		test to recover the image for machine vision tasks	
		