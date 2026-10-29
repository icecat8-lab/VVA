package com.vva.engine;

import android.graphics.Bitmap;
import android.hardware.HardwareBuffer;

import java.util.Arrays;

/**
 * Super-resolution engine for {@code .vva} models.
 *
 * <p>Typical use (call from a background thread, never the UI thread):
 *
 * <pre>
 * try (VVAEngine engine = new VVAEngine()) {
 *     if (!engine.init(modelFile.getAbsolutePath())) {
 *         Log.e("VVA", engine.getLastError());
 *         return;
 *     }
 *     Bitmap big = engine.upscale(photo, fraction -&gt; !cancelled);
 * }
 * </pre>
 *
 * <p>Two kinds of model are supported: image models ({@link #upscale}) and classifiers
 * ({@link #classify}), for example to choose between two upscaling models. {@link #isClassifier()}
 * tells which kind was loaded.
 *
 * <p>Processing runs on the CPU with several threads. Large images are split into
 * tiles automatically so memory stays within {@link #setMemoryBudgetMb(int)}.
 * {@link #cancel()} may be called from any thread.
 */
public final class VVAEngine implements AutoCloseable {

    static {
        System.loadLibrary("vva_engine");
    }

    /** Receives progress after every tile. Return {@code false} to cancel the job. */
    public interface ProgressListener {
        boolean onProgress(float fraction);
    }

    /** Result of {@link #classify}. */
    public static final class Classification {
        /** Most likely class. */
        public final int index;
        /** Its probability, 0..1. */
        public final float confidence;
        /** Class name from the model, or "class N" when the model has no names. */
        public final String label;
        /** One probability per class; they sum to 1. */
        public final float[] probabilities;

        Classification(int index, float confidence, String label, float[] probabilities) {
            this.index = index;
            this.confidence = confidence;
            this.label = label;
            this.probabilities = probabilities;
        }
    }

    private final Object runLock = new Object();     // one job / model load at a time
    private final Object handleLock = new Object();  // protects {@link #handle}
    private long handle;
    private volatile int scale;
    private volatile boolean classifier;
    private volatile String javaError;

    public VVAEngine() {
        handle = nativeCreate();
        if (handle == 0) {
            throw new IllegalStateException("cannot create the native engine (out of memory?)");
        }
    }

    /** Loads a {@code .vva} model from a file path. On failure the previous model stays active. */
    public boolean init(String modelPath) {
        synchronized (runLock) {
            long h = currentHandle();
            if (h == 0) {
                javaError = "the engine is closed";
                return false;
            }
            javaError = null;
            boolean ok = nativeInit(h, modelPath);
            if (ok) {
                scale = nativeScale(h);
                classifier = nativeIsClassifier(h);
            }
            return ok;
        }
    }

    /** Upscale factor of the loaded model (for example 4), or 0 when no model is loaded. */
    public int getScale() {
        return scale;
    }

    /** True when the loaded model is a classifier (use {@link #classify}); false for image models. */
    public boolean isClassifier() {
        return classifier;
    }

    /** Human readable reason for the last failure. */
    public String getLastError() {
        String local = javaError;
        if (local != null) {
            return local;
        }
        synchronized (handleLock) {
            return handle == 0 ? "the engine is closed" : nativeLastError(handle);
        }
    }

    /** Tile size in input pixels; 0 (default) chooses from the memory budget. */
    public void setTileSize(int pixels) {
        synchronized (handleLock) {
            if (handle != 0) nativeSetTileSize(handle, pixels);
        }
    }

    /** Overlap read around each tile in pixels (default 16). */
    public void setTilePad(int pixels) {
        synchronized (handleLock) {
            if (handle != 0) nativeSetTilePad(handle, pixels);
        }
    }

    /** Worker threads; 0 (default) picks automatically (up to 8). */
    public void setThreads(int threads) {
        synchronized (handleLock) {
            if (handle != 0) nativeSetThreads(handle, threads);
        }
    }

    /** Working memory for one tile in MB (default 320). Lower it on devices with little RAM. */
    public void setMemoryBudgetMb(int megabytes) {
        synchronized (handleLock) {
            if (handle != 0) nativeSetMemoryBudgetMb(handle, megabytes);
        }
    }

    /** Stops the running job as soon as possible. Safe to call from any thread. */
    public void cancel() {
        synchronized (handleLock) {
            if (handle != 0) nativeCancel(handle);
        }
    }

    /** Vulkan diagnostics only; processing does not need it. */
    public boolean hasVulkan() {
        synchronized (handleLock) {
            return handle != 0 && nativeHasVulkan(handle);
        }
    }

    public String getVulkanDeviceName() {
        synchronized (handleLock) {
            if (handle == 0) return "";
            String name = nativeVulkanDeviceName(handle);
            return name == null ? "" : name;
        }
    }

    /**
     * Upscales {@code src} and returns a new ARGB_8888 bitmap, or {@code null} on failure or
     * cancellation (see {@link #getLastError()}). Any bitmap configuration is accepted.
     */
    public Bitmap upscale(Bitmap src, ProgressListener listener) {
        if (src == null) {
            javaError = "bitmap is null";
            return null;
        }
        int factor = scale;
        if (factor <= 0) {
            javaError = classifier ? "this model is a classifier; use classify()" : "no model loaded";
            return null;
        }
        Bitmap in = src;
        boolean ownsIn = false;
        if (src.getConfig() != Bitmap.Config.ARGB_8888) {
            try {
                in = src.copy(Bitmap.Config.ARGB_8888, false);
            } catch (OutOfMemoryError e) {
                in = null;
            }
            if (in == null) {
                javaError = "cannot convert the bitmap to ARGB_8888";
                return null;
            }
            ownsIn = true;
        }
        Bitmap out = null;
        try {
            long outW = (long) in.getWidth() * factor;
            long outH = (long) in.getHeight() * factor;
            if (outW > 32767L || outH > 32767L || outW * outH * 4L > 1500000000L) {
                javaError = "the upscaled bitmap would be too large";
                return null;
            }
            try {
                out = Bitmap.createBitmap((int) outW, (int) outH, Bitmap.Config.ARGB_8888);
            } catch (OutOfMemoryError | IllegalArgumentException e) {
                javaError = "not enough memory for the output bitmap";
                return null;
            }
            if (processBitmap(in, out, listener)) {
                Bitmap result = out;
                out = null;
                return result;
            }
            return null;
        } finally {
            if (out != null) {
                out.recycle();
            }
            if (ownsIn) {
                in.recycle();
            }
        }
    }

    /**
     * Classifies a bitmap with a classifier model: the image is resized to the model's input size and
     * normalised as stored in the model file. Returns {@code null} on failure (see {@link #getLastError()}).
     * Any bitmap configuration is accepted. Call it from a background thread.
     */
    public Classification classify(Bitmap src) {
        if (src == null) {
            javaError = "bitmap is null";
            return null;
        }
        if (!classifier) {
            javaError = "the loaded model is not a classifier";
            return null;
        }
        Bitmap in = src;
        boolean ownsIn = false;
        if (src.getConfig() != Bitmap.Config.ARGB_8888) {
            try {
                in = src.copy(Bitmap.Config.ARGB_8888, false);
            } catch (OutOfMemoryError e) {
                in = null;
            }
            if (in == null) {
                javaError = "cannot convert the bitmap to ARGB_8888";
                return null;
            }
            ownsIn = true;
        }
        try {
            synchronized (runLock) {
                long h = currentHandle();
                if (h == 0) {
                    javaError = "the engine is closed";
                    return null;
                }
                javaError = null;
                float[] r = nativeClassifyBitmap(h, in, in.isPremultiplied());
                if (r == null || r.length < 3) {
                    return null;
                }
                int index = (int) r[0];
                String label = nativeClassLabel(h, index);
                return new Classification(index, r[1], label == null ? "class " + index : label,
                        Arrays.copyOfRange(r, 2, r.length));
            }
        } finally {
            if (ownsIn) {
                in.recycle();
            }
        }
    }

    /**
     * Upscales {@code in} into {@code out}. Both must be software ARGB_8888 bitmaps and
     * {@code out} must be exactly {@link #getScale()} times larger.
     */
    public boolean processBitmap(Bitmap in, Bitmap out, ProgressListener listener) {
        if (in == null || out == null) {
            javaError = "bitmap is null";
            return false;
        }
        synchronized (runLock) {
            long h = currentHandle();
            if (h == 0) {
                javaError = "the engine is closed";
                return false;
            }
            javaError = null;
            return nativeProcessBitmap(h, in, out, listener, in.isPremultiplied(), out.isPremultiplied());
        }
    }

    /**
     * Same as {@link #processBitmap} for {@link HardwareBuffer}s, reading and writing their
     * memory directly (no Bitmap or Java-heap copies). Both buffers must be single-layer
     * RGBA_8888; allocate the input with USAGE_CPU_READ_OFTEN and the output with
     * USAGE_CPU_WRITE_OFTEN. Colours are treated as non-premultiplied. This is CPU access,
     * not GPU zero-copy.
     */
    public boolean processZeroCopy(HardwareBuffer in, HardwareBuffer out, ProgressListener listener) {
        if (in == null || out == null) {
            javaError = "HardwareBuffer is null";
            return false;
        }
        synchronized (runLock) {
            long h = currentHandle();
            if (h == 0) {
                javaError = "the engine is closed";
                return false;
            }
            javaError = null;
            return nativeProcessBuffers(h, in, out, listener);
        }
    }

    public boolean processZeroCopy(HardwareBuffer in, HardwareBuffer out) {
        return processZeroCopy(in, out, null);
    }

    /** Cancels any running job and releases the native engine. Safe to call more than once. */
    @Override
    public void close() {
        cancel();
        synchronized (runLock) {
            long h;
            synchronized (handleLock) {
                h = handle;
                handle = 0;
            }
            if (h != 0) {
                nativeDestroy(h);
            }
            scale = 0;
        }
    }

    private long currentHandle() {
        synchronized (handleLock) {
            return handle;
        }
    }

    private static native long nativeCreate();

    private static native void nativeDestroy(long handle);

    private static native boolean nativeInit(long handle, String path);

    private static native String nativeLastError(long handle);

    private static native int nativeScale(long handle);

    private static native boolean nativeIsClassifier(long handle);

    private static native String nativeClassLabel(long handle, int index);

    private static native float[] nativeClassifyBitmap(long handle, Bitmap bitmap, boolean premultiplied);

    private static native void nativeSetTileSize(long handle, int pixels);

    private static native void nativeSetTilePad(long handle, int pixels);

    private static native void nativeSetThreads(long handle, int threads);

    private static native void nativeSetMemoryBudgetMb(long handle, int megabytes);

    private static native void nativeCancel(long handle);

    private static native boolean nativeHasVulkan(long handle);

    private static native String nativeVulkanDeviceName(long handle);

    private static native boolean nativeProcessBitmap(long handle, Bitmap in, Bitmap out, Object listener,
                                                      boolean inPremultiplied, boolean outPremultiplied);

    private static native boolean nativeProcessBuffers(long handle, HardwareBuffer in, HardwareBuffer out,
                                                       Object listener);
}
