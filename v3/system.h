#ifndef system_h
#define system_h

#define ratio 1.4
#define Pi 3.1415926535897932
#define sysdim 3

// type define
typedef struct
    {
    int x;
    int y;
    #if sysdim == 3
    int z;
    #endif
    } intv_t;

typedef struct
    {
    double x;
    double y;
    #if sysdim == 3
    double z;
    #endif
    } vec_t;

typedef struct
    {
    int     natom;
    double  phi;
    vec_t   len;
    vec_t   leninv;
    double  strain;
    } box_t;

typedef struct
    {
    int     seed;
    double  phi;
    } sets_t;

#endif
