#ifndef __SUPPORT_CU__
#define __SUPPORT_CU__

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "support.h"

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

#endif