from pathlib import Path

wf = Path(".github/workflows/android.yml")
if not wf.exists():
    raise SystemExit("Missing .github/workflows/android.yml")

s = wf.read_text()

build_block = '''      - name: Build release APK
        env:
          XTTS_KEYSTORE_PATH: ${{ runner.temp }}/xtts-release.jks
          XTTS_KEYSTORE_PASSWORD: ${{ secrets.XTTS_KEYSTORE_PASSWORD }}
          XTTS_KEY_ALIAS: xtts
          XTTS_KEY_PASSWORD: ${{ secrets.XTTS_KEY_PASSWORD }}
        run: gradle :app:assembleRelease

'''

assert build_block in s, "Build release APK block not found"

verify_block = build_block + '''      - name: Verify release APK package and signature
        run: |
          APK="app/build/outputs/apk/release/app-release.apk"
          test -s "$APK"
          echo "=== APK file ==="
          ls -lh "$APK"
          file "$APK"
          sha256sum "$APK"

          echo "=== ZIP alignment ==="
          zipalign -c -v 4 "$APK"

          echo "=== APK signature ==="
          apksigner verify --verbose --print-certs "$APK"

          echo "=== Package metadata ==="
          aapt dump badging "$APK" | head -n 20

      - name: Copy verified APK with explicit name
        run: |
          cp app/build/outputs/apk/release/app-release.apk \
             app/build/outputs/apk/release/XTTSv2-Android-verified.apk

'''

s = s.replace(build_block, verify_block, 1)

upload_old = '''      - uses: actions/upload-artifact@v4
        with:
          name: XTTSv2-Android
          path: app/build/outputs/apk/release/app-release.apk
'''

upload_new = '''      - uses: actions/upload-artifact@v4
        with:
          name: XTTSv2-Android-VERIFIED
          path: app/build/outputs/apk/release/XTTSv2-Android-verified.apk
          if-no-files-found: error
'''

assert upload_old in s, "Upload artifact block not found"
s = s.replace(upload_old, upload_new, 1)

wf.write_text(s)

print("APK verification workflow patch applied")
print("Adds zipalign verification")
print("Adds apksigner verify --verbose --print-certs")
print("Adds aapt package metadata dump")
print("Adds SHA-256 output")
print("Only verified APK is uploaded as XTTSv2-Android-VERIFIED")
