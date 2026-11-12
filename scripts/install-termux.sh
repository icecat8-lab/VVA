#!/bin/sh
set -eu

REPO="${VVA_REPO:-icecat8-lab/VVA}"
TAG="${VVA_TAG:-latest}"
PREFIX_DIR="${PREFIX:-$HOME/.local}"
BIN_DIR="$PREFIX_DIR/bin"

if [ "$(uname -o 2>/dev/null || true)" != "Android" ]; then
    echo "error: the prebuilt vva-cli is built for Android (Termux, adb shell)." >&2
    echo "On a Linux PC build it from source; see 'Building and testing' in the README." >&2
    exit 1
fi

case "$(uname -m)" in
    aarch64|arm64) ABI="arm64-v8a" ;;
    *)
        echo "error: the prebuilt vva-cli is for 64-bit ARM phones (arm64-v8a); this device reports $(uname -m)." >&2
        exit 1
        ;;
esac

if [ "$TAG" = "latest" ]; then
    BASE="https://github.com/$REPO/releases/latest/download"
else
    BASE="https://github.com/$REPO/releases/download/$TAG"
fi

for tool in curl sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "error: $tool is required (Termux: pkg install curl coreutils)" >&2
        exit 1
    }
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading vva-cli for $ABI ..."
curl -fL --retry 3 -o "$TMP/vva-cli-$ABI" "$BASE/vva-cli-$ABI"
curl -fL --retry 3 -o "$TMP/SHA256SUMS" "$BASE/SHA256SUMS"

EXPECTED="$(grep " vva-cli-$ABI\$" "$TMP/SHA256SUMS" | cut -d' ' -f1 || true)"
[ -n "$EXPECTED" ] || { echo "error: vva-cli-$ABI is not listed in SHA256SUMS" >&2; exit 1; }
ACTUAL="$(sha256sum "$TMP/vva-cli-$ABI" | cut -d' ' -f1)"
if [ "$EXPECTED" != "$ACTUAL" ]; then
    echo "error: checksum mismatch, the download is corrupted or was tampered with" >&2
    exit 1
fi

mkdir -p "$BIN_DIR"
install -m 755 "$TMP/vva-cli-$ABI" "$BIN_DIR/vva-cli"
echo "Installed $BIN_DIR/vva-cli"
"$BIN_DIR/vva-cli" --version || true

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) echo "Add it to your PATH:  export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac

echo
echo "Next:  vva-cli report your-model.vva   (prints a device report you can paste into a bug report)"
