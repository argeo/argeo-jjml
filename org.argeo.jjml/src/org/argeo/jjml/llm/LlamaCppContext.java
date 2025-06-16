package org.argeo.jjml.llm;

import static java.lang.System.Logger.Level.WARNING;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.System.Logger;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.function.LongSupplier;

import org.argeo.jjml.internal.OsUtils;
import org.argeo.jjml.llm.params.ContextParam;
import org.argeo.jjml.llm.params.ContextParams;
import org.argeo.jjml.llm.params.PoolingType;

/**
 * Access to a llama.cpp context
 * 
 * @see llama.h - llama_context
 */
public class LlamaCppContext implements LongSupplier, AutoCloseable {
	private final static Logger logger = System.getLogger(LlamaCppContext.class.getName());

	private final static ContextParams DEFAULT_CONTEXT_PARAMS_NATIVE;

	static {
		DEFAULT_CONTEXT_PARAMS_NATIVE = LlamaCppBackend.newContextParams();
	}

	private final long pointer;
	private final LlamaCppModel model;

	private final ContextParams initParams;

	// effective parameters
	private final PoolingType poolingType;
	private final int contextSize;
	private final int batchSize;
//	private final int physicalBatchSize;
	private final int maxSequenceCount;

//	private LlamaCppBatchProcessor batchProcessor;

	public LlamaCppContext(LlamaCppModel model) {
		this(model, DEFAULT_CONTEXT_PARAMS_NATIVE);
	}

	public LlamaCppContext(LlamaCppModel model, ContextParams initParams) {
		Objects.requireNonNull(model);
		Objects.requireNonNull(initParams);
		if (initParams.embeddings() && initParams.n_ubatch() != initParams.n_batch()) {
			initParams = initParams.with(ContextParam.n_batch, initParams.n_ubatch());
			logger.log(WARNING, "Embeddings requires same logical and physical batch size, forcing n_batch to "
					+ initParams.n_ubatch());
		}
		this.pointer = doInit(model, initParams);
		this.model = model;
		this.initParams = initParams;

		// effective parameters from native side
		int poolingTypeCode = doGetPoolingType();
		poolingType = PoolingType.byCode(poolingTypeCode);
		contextSize = doGetContextSize();

		batchSize = doGetBatchSize();
//		physicalBatchSize = doGetPhysicalBatchSize();
		maxSequenceCount = doGetMaxSequenceCount();
	}

	/*
	 * NATIVE
	 */
	private static native long doInit(LlamaCppModel model, ContextParams params);

	private native void doDestroy();

	private native int doGetPoolingType();

	private native int doGetContextSize();

	private native int doGetBatchSize();

	private native int doGetPhysicalBatchSize();

	private native int doGetMaxSequenceCount();

	private native long doGetStateSize();

	private native byte[] doGetStateDataAsBytes();

	private native int doGetStateData(ByteBuffer buf, int offset);

	private native void doSetStateDataBytes(byte[] arr, int offset, int length);

	private native void doSetStateData(ByteBuffer buf, int offset, int length);

	private native void doSaveStateFile(byte[] path, IntBuffer buf, int offset, int length);

	private native int doLoadStateFile(byte[] path, IntBuffer buf, int offset);

	/*
	 * STATE
	 */
	long getStateSize() {
		return doGetStateSize();
	}

	void readState(ByteBuffer buf) {
		if (buf.isDirect()) {
			int offset = buf.position();
			int read = doGetStateData(buf, offset);
			buf.position(offset + read);
		} else {
			byte[] arr = doGetStateDataAsBytes();
			buf.put(arr);
		}
	}

	void writeState(ByteBuffer buf) {
		if (buf.isDirect()) {
			doSetStateData(buf, 0, buf.limit());
			buf.position(buf.limit());
		} else {
			byte[] arr = buf.array();
			doSetStateDataBytes(arr, 0, arr.length);
		}
	}

	void saveStateFile(Path path, IntBuffer tokens) throws IOException {
		if (!tokens.isDirect())
			throw new IllegalArgumentException("Tokens must be in a direct buffer");
		if (Files.exists(path) && !Files.isWritable(path))
			throw new IOException("Location " + path + " for session file is not writable");
		doSaveStateFile(OsUtils.filePathToNative(path), tokens, 0, tokens.position());
	}

	int loadStateFile(Path path, IntBuffer tokens) throws IOException {
		if (!Files.exists(path))
			throw new FileNotFoundException("Session file " + path + " does not exist");
		if (!tokens.isDirect())
			throw new IllegalArgumentException("Tokens must be in a direct buffer");
		int tokenCount = doLoadStateFile(OsUtils.filePathToNative(path), tokens, 0);
		tokens.position(tokenCount);
		return tokenCount;
	}

	/*
	 * LIFECYCLE
	 */
	@Override
	public void close() throws RuntimeException {
		doDestroy();
	}

	/*
	 * PACKAGE COORDINATION
	 */
//	void setBatchProcessor(LlamaCppBatchProcessor batchProcessor) {
//		if (batchProcessor != null)
//			throw new IllegalArgumentException("A batch processor is already active for this context");
//		this.batchProcessor = batchProcessor;
//	}

	/*
	 * ACCESSORS
	 */
	@Override
	public long getAsLong() {
		return pointer;
	}

	public LlamaCppModel getModel() {
		return model;
	}

	public ContextParams getInitParams() {
		return initParams;
	}

	public PoolingType getPoolingType() {
		return poolingType;
	}

	public int getContextSize() {
		return contextSize;
	}

	public int getBatchSize() {
		return batchSize;
	}

//	public LlamaCppBatchProcessor getBatchProcessor() {
//		return batchProcessor;
//	}

//	public int getPhysicalBatchSize() {
//		return physicalBatchSize;
//	}

	public int getMaxSequenceCount() {
		return maxSequenceCount;
	}

	/*
	 * STATIC UTILTIES
	 */
	public static ContextParams defaultContextParams() {
		ContextParams res = DEFAULT_CONTEXT_PARAMS_NATIVE;
		for (ContextParam param : ContextParam.values()) {
			String sysProp = System.getProperty(param.asSystemProperty());
			if (sysProp != null)
				res = res.with(param, sysProp);
		}
		return res;
	}
}
