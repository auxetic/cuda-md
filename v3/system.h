#ifndef system_h
#define system_h

#define ratio 1.4
#define Pi 3.1415926535897932
#define sysdim 3

// type define
typedef struct intv_t
    {
    int x;
    int y;
    #if sysdim == 3
    int z;
    #endif

    } intv_t;

typedef struct vec_t
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

//#ifndef __operator_plus_intv_t
//intv_t operator+( const intv_t lhs, const intv_t rhs )
//    {
//    intv_t sum;
//    sum.x = lhs.x + rhs.x;
//    sum.y = lhs.y + rhs.y;
//    sum.z = lhs.z + rhs.z;
//    return sum;
//    }
//#endif
//
//#ifndef __operator_plus_vec_t
//vec_t operator+( const vec_t lhs, const vec_t rhs )
//    {
//    vec_t sum;
//    sum.x = lhs.x + rhs.x;
//    sum.y = lhs.y + rhs.y;
//    sum.z = lhs.z + rhs.z;
//    return sum;
//    }
//#endif

#endif
