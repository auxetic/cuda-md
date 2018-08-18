#ifndef system_h
#define system_h

#define ratio   1.4
#define Pi      3.1415926535897932
#define sysdim  2

#define const_32   32
#define const_64   64
#define const_128  128
#define const_256  256
#define const_512  512
#define const_1024 1024
#define const_2048 2048

// type define
typedef struct
    {
    int x;
    int y;
    #if sysdim == 3
        int z;
    #endif
    } intd;

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
    } tpbox;

typedef struct
    {
    int     seed;
    double  phi;
    } tpsets;

// variables define
static tpbox  box;
static tpsets sets;

#endif
