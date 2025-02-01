#include <iostream>
#include <thread>
using namespace std;
typedef unsigned int cu[32];
void hostRandomGenerationTest()
{
    clock_t start, end;
    cout << "--------Random Number Generation Test--------" << endl;
    for (int j = 1; j <= 8; j++)
    {
        float* a = new float[100000000];
        for (int i = 0; i < 100000000; i++)
            a[i] = float(rand()) / RAND_MAX;
        delete[] a;
    }
    cout << "0% Done";
    start = clock();
    for (int j = 1; j <= 100; j++)
    {
        float* a = new float[100000000];
        for (int i = 0; i < 100000000; i++)
            a[i] = float(rand()) / RAND_MAX;
        delete[] a;
        if (j % 10 == 0)
            cout << endl << j / 1 << "% Done";
        if (j % 1 == 0)
            cout << ".";
    }
    end = clock();
    cout << "Time: " << (double)(end - start) / CLOCKS_PER_SEC / 100 << "s" << endl;
    cout << "--------Random Number Generation Test Done--------" << endl;
    cout << "--------Random Number Generation Test (Multi thread) --------" << endl;
    auto f = [&](float* a, const int offset)
    {
        for (int i = 0; i < 1000000; i++)
            a[offset + i] = float(rand()) / RAND_MAX;
    };
    start = clock();
    for (int j = 1; j <= 100; j++)
    {
        float* a = new float[100000000];
        thread t[1000];
        for (int i = 0; i < 100; i++)
            t[i] = thread(f, a, i * 1000000);
        for (int i = 0; i < 100; i++)
            t[i].join();
        delete[] a;
        if (j % 10 == 0)
            cout << endl << j / 1 << "% Done";
        if (j % 1 == 0)
            cout << ".";
    }
    end = clock();
    cout << "Time: " << (double)(end - start) / CLOCKS_PER_SEC / 1000 << "s" << endl;
    cout << "--------Random Number Generation Test (Multi thread) Done--------" << endl;
    return;
}

int main()
{
    cu* a = new cu[32];
    delete[] a;
    return 0;
}