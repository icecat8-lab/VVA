package com.vva.engine;

import android.graphics.Bitmap;
import android.hardware.HardwareBuffer;

import java.util.Arrays;

public final class VVAEngine implements AutoCloseable {

    static {
        System.loadLibrary("vva_engine");
    }

    public static final int DEVICE_AUTO = 0;
    public static final int DEVICE_CPU = 1;
    public static final int DEVICE_GPU = 2;

    public interface ProgressListener {
        boolean onProgress(float fraction);
    }

    public static final class Classification {
        public final int index;
        public final float confidence;
        public final String label;
        public final float[] probabilities;

        Classification(int index, float confidence, String label, float[] probabilities) {
            this.index = index;
            this.confidence = confidence;
            this.label = label;
            this.probabilities = probabilities;
        }
    }

    private final Object runLock = new Object();
    private final Object handleLock = new Object();
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

    public int getScale() {
        return scale;
    }

    public void setDevice(int device) {
        synchronized (handleLock) {
            if (handle != 0) nativeSetDevice(handle, device);
        }
    }

    public int getLastDevice() {
        synchronized (handleLock) {
            return handle == 0 ? DEVICE_CPU : nativeLastDevice(handle);
        }
    }

    public String getGpuStatus() {
        synchronized (handleLock) {
            if (handle == 0) return "the engine is closed";
            String s = nativeGpuStatus(handle);
            return s == null ? "" : s;
        }
    }

    public boolean isClassifier() {
        return classifier;
    }

    public String getLastError() {
        String local = javaError;
        if (local != null) {
            return local;
        }
        synchronized (handleLock) {
            return handle == 0 ? "the engine is closed" : nativeLastError(handle);
        }
    }

    public void setTileSize(int pixels) {
        synchronized (handleLock) {
            if (handle != 0) nativeSetTileSize(handle, pixels);
        }
    }

    public void setTilePad(int pixels) {
        synchronized (handleLock) {
            if (handle != 0) nativeSetTilePad(handle, pixels);
        }
    }

    public void setThreads(int threads) {
        synchronized (handleLock) {
            if (handle != 0) nativeSetThreads(handle, threads);
        }
    }

    public void setMemoryBudgetMb(int megabytes) {
        synchronized (handleLock) {
            if (handle != 0) nativeSetMemoryBudgetMb(handle, megabytes);
        }
    }

    public void cancel() {
        synchronized (handleLock) {
            if (handle != 0) nativeCancel(handle);
        }
    }

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

    private static native void nativeSetDevice(long handle, int device);

    private static native int nativeLastDevice(long handle);

    private static native String nativeGpuStatus(long handle);

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
