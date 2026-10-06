/* Runs cml_selftest() at every ISA level the host supports; exit code 0 when all pass. */
#include "cml_kernels.h"
#include <stdio.h>

int main(void) {
    int fail = 0;
    for (int lvl = 0; lvl < 32; lvl++) {
        if (cml_set_isa(lvl) != lvl) continue;
        int rc = cml_selftest();
        printf("%-13s %s (rc=%d)\n", cml_isa_name(lvl), rc ? "FAIL" : "ok", rc);
        if (rc) fail = 1;
    }
    return fail;
}
