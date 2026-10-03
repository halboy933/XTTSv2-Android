from pathlib import Path

wf = Path(".github/workflows/android.yml")
if not wf.exists():
    raise SystemExit("Missing .github/workflows/android.yml")

s = wf.read_text()

build_line = '        run: gradle :app:assembleRelease\n'
upload_marker = '      - uses: actions/upload-artifact@v4\n'

if build_line not in s:
    raise SystemExit("Build release command not found")
if upload_marker not in s:
    raise SystemExit("Upload artifact marker not found")

if "Verify release APK package and signature" in s:
    print("APK verification block already present; nothing to patch.")
    raise SystemExit(0)

verify = '''
      - name: Verify release APK package and signature
        run: |
          APK="app/build/outputs/apk/release/app-release.apk"
          BT="$ANDROID_HOME/build-tools/36.0.0"

          test -s "$APK"

          echo "=== APK file ==="
          ls -lh "$APK"
          file "$APK"
          sha256sum "$APK"

          echo "=== ZIP alignment ==="
          "$BT/zipalign" -c -v 4 "$APK"

          echo "=== APK signature ==="
          "$BT/apksigner" verify --verbose --print-certs "$APK"

          echo "=== Package metadata ==="
          "$BT/aapt" dump badging "$APK" | head -n 20

      - name: Copy verified APK with explicit name
        run: |
          cp app/build/outputs/apk/release/app-release.apk \
             app/build/outputs/apk/release/XTTSv2-Android-verified.apk

'''

s = s.replace(build_line, build_line + verify, 1)

before, _after = s.split(upload_marker, 1)

new_upload = '''      - uses: actions/upload-artifact@v4
        with:
          name: XTTSv2-Android-VERIFIED
          path: app/build/outputs/apk/release/XTTSv2-Android-verified.apk
          if-no-files-found: error
'''

s = before + new_upload
wf.write_text(s)

print("APK verification workflow patch applied successfully")
print("Uses Build Tools 36.0.0 explicitly")
print("Checks: file / sha256 / zipalign / apksigner / aapt")
print("Artifact name: XTTSv2-Android-VERIFIED")
