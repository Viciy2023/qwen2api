import asyncio
import os
from contextlib import asynccontextmanager
from backend.core.config import settings

_CAMOUFOX_OPTS = {
    "headless": True,
    "humanize": False,
    "i_know_what_im_doing": True,
    "firefox_user_prefs": {
        "layers.acceleration.disabled": True,
        "gfx.webrender.enabled": False,
        "gfx.webrender.all": False,
        "gfx.webrender.software": False,
        "gfx.canvas.azure.backends": "skia",
        "media.hardware-video-decoding.enabled": False,
    },
}


@asynccontextmanager
async def _new_browser():
    if settings.DISABLE_BROWSER_AUTOMATION:
        raise RuntimeError("Browser automation is disabled in this deployment")

    from camoufox.async_api import AsyncCamoufox

    async with AsyncCamoufox(**_CAMOUFOX_OPTS) as browser:
        yield browser


async def ensure_browser_installed():
    if settings.DISABLE_BROWSER_AUTOMATION:
        return

    import subprocess
    import sys

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                [sys.executable, "-m", "camoufox", "path"],
                capture_output=True,
                text=True,
                timeout=10,
            ),
        )
        cache_dir = result.stdout.strip()
        if cache_dir:
            exe_name = "camoufox.exe" if os.name == "nt" else "camoufox"
            exe_path = os.path.join(cache_dir, exe_name)
            if os.path.exists(exe_path):
                return
    except Exception:
        pass

    loop = asyncio.get_event_loop()

    def _do_install():
        from camoufox.pkgman import CamoufoxFetcher

        CamoufoxFetcher().install()

    await loop.run_in_executor(None, _do_install)
