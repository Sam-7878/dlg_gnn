// Exact upper-tail count for tie-free Friedman blocks. The first block is fixed
// by model-label symmetry. Sorting partial sums is a state canonicalization;
// each possible assignment of the next block still contributes its multiplicity.
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <numeric>
#include <unordered_map>
#include <vector>
using Sum = std::array<int,7>;
using Count = unsigned __int128;
int n,k,threshold;
std::vector<Sum> permutations;
std::vector<std::unordered_map<uint64_t, Count>> memo;
std::vector<Count> powers;

Count tail(Sum sums, int left) {
    int score=0, maximum=0;
    uint64_t key=0;
    for(int i=0;i<k;++i) {
        score+=sums[i]*sums[i];
        int aligned=sums[i]+left*(i+1);
        maximum+=aligned*aligned;
        key=(key<<7)|sums[i];
    }
    if(maximum<threshold) return 0;
    if(left==0) return score>=threshold;
    auto &cache=memo[left];
    if(auto it=cache.find(key);it!=cache.end()) return it->second;
    Count count=0;
    for(auto &rank:permutations) {
        Sum next{};
        for(int i=0;i<k;++i) next[i]=sums[i]+rank[i];
        if(left==1) {
            int q=0;
            for(int i=0;i<k;++i) q+=next[i]*next[i];
            count+=(q>=threshold);
        } else {
            std::sort(next.begin(),next.begin()+k);
            count+=tail(next,left-1);
        }
    }
    cache.emplace(key,count);
    return count;
}

void show(Count value) {
    std::string digits;
    do { digits.push_back('0'+value%10);value/=10; } while(value);
    std::reverse(digits.begin(),digits.end());
    std::cout<<digits;
}

int main() {
    while(std::cin>>n>>k>>threshold) {
        Sum first{};
        std::iota(first.begin(),first.begin()+k,1);
        Sum rank=first;
        permutations.clear();
        do { permutations.push_back(rank); } while(std::next_permutation(rank.begin(),rank.begin()+k));
        memo.clear();
        memo.resize(n);
        Count total=1;
        for(int i=1;i<n;++i) total*=permutations.size();
        auto favorable=tail(first,n-1);
        show(favorable);std::cout<<" ";show(total);std::cout<<std::endl;
    }
}
