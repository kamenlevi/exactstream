/* records3 layout guard (MoonShots, Wed 2026-11-04; Leap's triage ~/Leap/data/records3_readers_triage.md): the SSD micro
   harnesses (ssdmicro*.c / .m) assume today's row-major records3 (record stride, part sizes 2,162,688 / 770,048 B). The
   in-place native+p11+pad conversion (Thu) adds a per-layer "layout" key to index.json; refuse to run once any layer
   carries a layout other than row-major, so no timing silently uses the old layout's sizes. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void records3_guard(void) {
  char path[512];
  snprintf(path, sizeof path, "%s/BigMoE/models/bare/records3/index.json", getenv("HOME"));
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(3); }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *s = (char *)malloc((size_t)n + 1);
  if (!s || fread(s, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "records3_guard: cannot read %s\n", path); exit(3); }
  s[n] = 0;
  fclose(f);
  for (char *p = strstr(s, "\"layout\""); p; p = strstr(p + 8, "\"layout\"")) {
    char *v = p + 8;
    while (*v == ' ' || *v == ':' || *v == '\t' || *v == '\n' || *v == '\r') v++;
    if (*v == '"') v++;
    if (strncmp(v, "row", 3) && *v != '0') {
      fprintf(stderr, "records3_guard: %s has a converted layer (layout \"%.20s...\"): this harness assumes row-major records; "
              "update its part sizes and offsets before timing\n", path, v);
      exit(3);
    }
  }
  free(s);
}
