/* sbcheck: is a process's sandbox fingerprint the agent-test task profile's?
   usage: sbcheck <pid> <path that must be writable> <path that must be denied> [more denied paths...]
   prints "<sandboxed> <allowed-write?> <denied...>" and exits 0 if the process is sandboxed, may create files under the
   first path and may not under every other path; exit 1 if not; exit 2 on a usage error. */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
enum { SB_FILTER_NONE = 0, SB_FILTER_PATH = 1 };
#define SB_NO_REPORT 0x40000000
int sandbox_check(pid_t pid, const char *operation, int type, ...);
int main(int argc, char **argv) {
    if (argc < 4) return 2;
    pid_t pid = (pid_t)atoi(argv[1]);
    int sb = sandbox_check(pid, NULL, SB_FILTER_NONE);
    int allow = sandbox_check(pid, "file-write-create", SB_FILTER_PATH | SB_NO_REPORT, argv[2]);
    int ok = (sb == 1) && (allow == 0);
    printf("%d %d", sb, allow);
    for (int i = 3; i < argc; i++) {
        int d = sandbox_check(pid, "file-write-create", SB_FILTER_PATH | SB_NO_REPORT, argv[i]);
        printf(" %d", d);
        if (d != 1) ok = 0;
    }
    printf("\n");
    return ok ? 0 : 1;
}
