get_bsNsvg_v2.py
	The encoder and decoder components.
	The program scan the provided folders to produce the bit-streams. For each image and the corresponding SVG file produced by the AutoTrace tool, the program genetrate the quantized vectorized representation (TXT), the enhancement bitstream consisting of the RGB color samples (RGB) and the re-rendered edges/color samples and masks (PNG).

	parse_svg():
		To parse an SVG file to make the compact vectorized representation. This is the main encoder part.

	get_ref_point_line():
		Get RGB samples according to segments.

	get_bezier_inner_point():
		Get RGB samples according to curves.

	handle_image():
		Make the bit-stream of an image with the provided SVG file, image file, and make the above mentioned three outputs.

quantize.py
	Quantize the SVG parameters and make the compact representation.

make_ppm_parallel.py
	In the parallel way, losslessly compress the compact text representation with the PPM algorithm provided by 7zip tools.