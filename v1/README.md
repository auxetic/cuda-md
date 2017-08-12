## GPU 分子动力学&能量最小化程序模板

### 说明
* 需要在配置好CUDA环境的服务器上编译使用
* v1版本。使用列表法加速计算，列表是会进行空间分块，从而使算法复杂度降为O(N)。
* 算法有显而易见的访寸问题。

```bash
cp templates/xx.cu main.cu
mkdir build && cd build
cmake ..
make
```
