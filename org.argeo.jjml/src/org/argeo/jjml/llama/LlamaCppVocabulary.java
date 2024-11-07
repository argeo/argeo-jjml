package org.argeo.jjml.llama;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;

public class LlamaCppVocabulary {
	private final LlamaCppModel model;

	public LlamaCppVocabulary(LlamaCppModel model) {
		super();
		this.model = model;
	}

	// Tokenization
	/**
	 * Tokenize a string encoded as a standard UTF-8 byte array. To use when it
	 * makes more sense to convert on the Java side.
	 * 
	 * @see #doDeTokenizeAsUtf8Array(int[], boolean, boolean)
	 */
	private native int[] doTokenizeUtf8BytesAsArray(long pointer, byte[] str, int offset, int length,
			boolean addSpecial, boolean parseSpecial);

	private native int[] doTokenizeUtf8AsArray(long pointer, ByteBuffer str, boolean addSpecial, boolean parseSpecial);

	private native void doTokenizeUtf8Bytes(long pointer, byte[] str, int offset, int length, IntBuffer tokens,
			boolean addSpecial, boolean parseSpecial);

	private native void doTokenizeUtf8(long pointer, ByteBuffer str, IntBuffer tokens, boolean addSpecial,
			boolean parseSpecial);

	/** De-tokenize as a string encoded in standard UTF-8. */
	private native byte[] doDeTokenizeArrayAsUtf8Bytes(long pointer, int[] tokens, int offset, int length,
			boolean removeSpecial, boolean unparseSpecial);

	private native byte[] doDeTokenizeAsUtf8Bytes(long pointer, IntBuffer tokens, boolean removeSpecial,
			boolean unparseSpecial);

	private native void doDeTokenizeArrayAsUtf8(long pointer, int[] tokens, int offset, int length, ByteBuffer str,
			boolean removeSpecial, boolean unparseSpecial);

	private native void doDeTokenizeAsUtf8(long pointer, IntBuffer tokens, ByteBuffer str, boolean removeSpecial,
			boolean unparseSpecial);

	/**
	 * Tokenize a Java {@link String}. Its UTF-16 representation will be used
	 * without copy on the native side, where it will be converted to UTF-8.
	 */
	native int[] doTokenizeStringAsArray(long pointer, String str, boolean addSpecial, boolean parseSpecial);

	private native void doTokenizeString(long pointer, String str, IntBuffer buf, boolean addSpecial,
			boolean parseSpecial);

	/** De-tokenize as a Java {@link String}. */
	native String doDeTokenizeArrayAsString(long pointer, int[] tokens, int offset, int length, boolean removeSpecial,
			boolean unparseSpecial);

	private native String doDeTokenizeAsString(long pointer, IntBuffer buf, boolean removeSpecial,
			boolean unparseSpecial);

	/*
	 * UTF-8
	 */
	public void tokenizeUtf8(ByteBuffer str, IntBuffer tokens, boolean addSpecial, boolean parseSpecial) {
		synchronized (tokens) {// we are writing into this buffer and changing its position
			// make sure position is 0
			ByteBuffer strToUse = str.slice().limit(str.limit() - str.position());
			IntBuffer tokensToUse = tokens.slice().limit(tokens.limit() - tokens.position());

			if (tokensToUse.isDirect()) {
				if (strToUse.isDirect()) {
					doTokenizeUtf8(model.getAsLong(), strToUse, tokensToUse, addSpecial, parseSpecial);
				} else if (strToUse.hasArray()) {
					byte[] arr = strToUse.array();
					doTokenizeUtf8Bytes(model.getAsLong(), arr, strToUse.arrayOffset(), strToUse.limit(), tokensToUse,
							addSpecial, parseSpecial);
				} else {
					throw new IllegalArgumentException("UTF-8 buffer is neither direct nor array-backed");
				}
				tokens.position(tokens.position() + tokensToUse.position());
			} else {
				int[] tokenArr;
				if (strToUse.isDirect()) {
					tokenArr = doTokenizeUtf8AsArray(model.getAsLong(), strToUse, addSpecial, parseSpecial);
				} else if (strToUse.hasArray()) {
					byte[] arr = strToUse.array();
					tokenArr = doTokenizeUtf8BytesAsArray(model.getAsLong(), arr, strToUse.arrayOffset(),
							strToUse.limit(), addSpecial, parseSpecial);
				} else {
					throw new IllegalArgumentException("UTF-8 buffer is neither direct nor array-backed");
				}
				if (tokenArr.length > (tokens.limit() - tokens.position()))
					throw new IndexOutOfBoundsException(tokenArr.length);
				tokens.put(tokenArr);
			}
		}
	}

	/*
	 * UTF-16
	 */
	public void tokenizeUtf16(String str, IntBuffer tokens, boolean addSpecial, boolean parseSpecial) {
		synchronized (tokens) {// we are writing into this buffer and changing its position
			IntBuffer tokensToUse = tokens.slice().limit(tokens.limit() - tokens.position());
			if (tokens.isDirect()) {
				doTokenizeString(model.getAsLong(), str, tokensToUse, addSpecial, parseSpecial);
				tokens.position(tokens.position() + tokensToUse.position());
			} else {
				int[] tokenArr = doTokenizeStringAsArray(model.getAsLong(), str, addSpecial, parseSpecial);
				if (tokenArr.length > (tokens.limit() - tokens.position()))
					throw new IndexOutOfBoundsException(tokenArr.length);
				tokens.put(tokenArr);
			}
		}
	}


}
