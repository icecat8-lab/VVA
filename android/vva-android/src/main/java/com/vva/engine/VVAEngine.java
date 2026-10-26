package com.vva.engine;

import android.graphics.Bitmap;
import android.hardware.HardwareBuffer;

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

    private final Object runLock = new Object();     // one job / model load at a time
    private final Object handleLock = new Object();  // protects {@link #handle}
    private long handle;
    private volatile int scale;
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
            }
            return ok;
        }
    }

    /** Upscale factor of the loaded model (for example 4), or 0 when no model is loaded. */
    public int getScale() {
        return scale;
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
            javaError = "no model loaded";
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
