package com.vva.engine;
import android.hardware.HardwareBuffer;

public class VVAEngine {
    static { System.loadLibrary("vva_engine"); }
    public boolean init(String path) { return nativeInit(path); }
    private native boolean nativeInit(String path);
}
