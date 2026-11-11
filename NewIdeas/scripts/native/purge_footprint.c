// Does volatile purgeable memory count in phys_footprint (the RAM-budget metric)? 256 MB, touched, toggled.
#include <stdio.h>
#include <string.h>
#include <mach/mach.h>
#include <mach/vm_purgable.h>
#include <libproc.h>
#include <sys/resource.h>
#include <unistd.h>
static unsigned long long fp(void) {
    task_vm_info_data_t info; mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &n); return info.phys_footprint;
}
static unsigned long long lifetime(void) {
    struct rusage_info_v4 ri; proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&ri); return ri.ri_lifetime_max_phys_footprint;
}
int main(void) {
    const vm_size_t SZ = 256ull << 20; vm_address_t a = 0;
    printf("start            fp %6.1f MB  lifetime %6.1f MB\n", fp() / 1048576.0, lifetime() / 1048576.0);
    if (vm_allocate(mach_task_self(), &a, SZ, VM_FLAGS_ANYWHERE | VM_FLAGS_PURGABLE) != KERN_SUCCESS) { puts("alloc failed"); return 1; }
    memset((void *)a, 0x5a, SZ);
    printf("touched (nonvol) fp %6.1f MB  lifetime %6.1f MB\n", fp() / 1048576.0, lifetime() / 1048576.0);
    int st = VM_PURGABLE_VOLATILE; vm_purgable_control(mach_task_self(), a, VM_PURGABLE_SET_STATE, &st);
    printf("volatile         fp %6.1f MB  lifetime %6.1f MB\n", fp() / 1048576.0, lifetime() / 1048576.0);
    st = VM_PURGABLE_NONVOLATILE; vm_purgable_control(mach_task_self(), a, VM_PURGABLE_SET_STATE, &st);
    printf("nonvolatile again fp %6.1f MB  lifetime %6.1f MB  (previous state %s)\n", fp() / 1048576.0, lifetime() / 1048576.0,
           (st & VM_PURGABLE_EMPTY) ? "EMPTY (purged)" : "kept");
    vm_address_t b = 0;                                   // allocate volatile from the start, then touch: never charged?
    vm_allocate(mach_task_self(), &b, SZ, VM_FLAGS_ANYWHERE | VM_FLAGS_PURGABLE);
    st = VM_PURGABLE_VOLATILE; vm_purgable_control(mach_task_self(), b, VM_PURGABLE_SET_STATE, &st);
    memset((void *)b, 0x3c, SZ);
    printf("2nd buffer written while volatile: fp %6.1f MB  lifetime %6.1f MB\n", fp() / 1048576.0, lifetime() / 1048576.0);
    return 0;
}
