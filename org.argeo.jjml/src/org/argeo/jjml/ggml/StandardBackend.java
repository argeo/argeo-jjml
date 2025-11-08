package org.argeo.jjml.ggml;

/** Standard GGML backends. */
public enum StandardBackend {
	cpu, //
	vulkan, //
	cuda, //
	hip, //
	blas, //
	rpc, //
	// unsupported:
	cann, //
	metal, //
	sycl, //
	opencl, //
	musa, //
	;
}
