class IdMap:
    """
    Ingat kembali di kuliah, bahwa secara praktis, sebuah dokumen dan
    sebuah term akan direpresentasikan sebagai sebuah integer. Oleh
    karena itu, kita perlu maintain mapping antara string term (atau
    dokumen) ke integer yang bersesuaian, dan sebaliknya. Kelas IdMap ini
    akan melakukan hal tersebut.
    """

    def __init__(self):
        """
        Mapping dari string (term atau nama dokumen) ke id disimpan dalam
        python's dictionary; cukup efisien. Mapping sebaliknya disimpan dalam
        python's list.

        contoh:
            str_to_id["halo"] ---> 8
            str_to_id["/collection/dir0/gamma.txt"] ---> 54

            id_to_str[8] ---> "halo"
            id_to_str[54] ---> "/collection/dir0/gamma.txt"
        """
        self.str_to_id = {}
        self.id_to_str = []

    def __len__(self):
        """Mengembalikan banyaknya term (atau dokumen) yang disimpan di IdMap."""
        return len(self.id_to_str)

    def __get_str(self, i):
        """Mengembalikan string yang terasosiasi dengan index i."""
        # TODO
        str_elem = self.id_to_str[i]
        return str_elem

    def __get_id(self, s):
        """
        Mengembalikan integer id i yang berkorespondensi dengan sebuah string s.
        Jika s tidak ada pada IdMap, lalu assign sebuah integer id baru dan kembalikan
        integer id baru tersebut.
        """
        # TODO
        id_str = 0
        if s in self.str_to_id:
            id_str = self.str_to_id[s]
        else:
            self.str_to_id[s] = len(self.str_to_id)
            id_str = self.str_to_id[s]
            self.id_to_str.append(s)
        # print("S ",s)
        # print("str_to_id ",self.str_to_id)
        # print("id to str ",self.id_to_str)
        # print("id_str ",id_str)
        return id_str

    def __getitem__(self, key):
        """
        __getitem__(...) adalah special method di Python, yang mengizinkan sebuah
        collection class (seperti IdMap ini) mempunyai mekanisme akses atau
        modifikasi elemen dengan syntax [..] seperti pada list dan dictionary di Python.

        Silakan search informasi ini di Web search engine favorit Anda. Saya mendapatkan
        link berikut:

        https://stackoverflow.com/questions/43627405/understanding-getitem-method

        Jika key adalah integer, gunakan __get_str;
        jika key adalah string, gunakan __get_id
        """
        if type(key) is int:
            return self.__get_str(key)
        elif type(key) is str:
            return self.__get_id(key)
        else:
            raise TypeError

def sorted_intersect(list1, list2):
    """
    Intersects two (ascending) sorted lists and returns the sorted result
    Melakukan Intersection dua (ascending) sorted lists dan mengembalikan hasilnya
    yang juga terurut.

    Parameters
    ----------
    list1: List[Comparable]
    list2: List[Comparable]
        Dua buah sorted list yang akan di-intersect.

    Returns
    -------
    List[Comparable]
        intersection yang sudah terurut
    """
    # TODO
    answer = []
    counter1 = 0
    counter2 = 0
    # print("#######################")
    # print(list1)
    # print(list2)
    while list1 and list2 and counter1 < len(list1) and counter2 < len(list2) :
        # print("Counter 1 :", counter1)
        # print("Counter 2 :", counter2)
        if list1[counter1] == list2[counter2]:
            answer.append(list1[counter1])
            counter1+=1
            counter2+=1
        elif list1[counter1] < list2[counter2]:
            counter1+=1
        else:
            counter2+=1
        # print("bawah")
        # print("Counter 1 :", counter1)
        # print("Counter 2 :", counter2)
        # print("bawah")
    # print("ans:")
    # print(answer)
    
    return answer

if __name__ == '__main__':

    doc = ["halo", "semua", "selamat", "pagi", "semua"]
    term_id_map = IdMap()
    assert [term_id_map[term] for term in doc] == [0, 1, 2, 3, 1], "term_id salah"
    assert term_id_map[1] == "semua", "term_id salah"
    assert term_id_map[0] == "halo", "term_id salah"
    assert term_id_map["selamat"] == 2, "term_id salah"
    assert term_id_map["pagi"] == 3, "term_id salah"

    docs = ["/collection/0/data0.txt",
            "/collection/0/data10.txt",
            "/collection/1/data53.txt"]
    doc_id_map = IdMap()
    assert [doc_id_map[docname] for docname in docs] == [0, 1, 2], "docs_id salah"
    
    assert sorted_intersect([1, 2, 3], [2, 3]) == [2, 3], "sorted_intersect salah"
    assert sorted_intersect([4, 5], [1, 4, 7]) == [4], "sorted_intersect salah"
    assert sorted_intersect([], []) == [], "sorted_intersect salah"
