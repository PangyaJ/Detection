import os
from tkinter import filedialog, Tk
import shutil

def delete_small_files(folder_path):
    # 1KB = 1024 bytes
    size_limit = 1024
    
    # 폴더 내 모든 하위 폴더와 파일 순회
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # 파일 크기 확인
                file_size = os.path.getsize(file_path)
                # 1KB 미만인 경우 삭제
                if file_size < size_limit:
                    os.remove(file_path)
                    print(f"삭제됨: {file_path} ({file_size} bytes)")
            except Exception as e:
                print(f"오류 발생 - {file_path}: {str(e)}")

def main():
    # GUI 창 생성 (보이지 않게)
    root = Tk()
    root.withdraw()  # 메인 윈도우 숨기기
    
    # 폴더 선택 다이얼로그 표시
    selected_folder = filedialog.askdirectory(
        title="삭제할 파일이 있는 폴더를 선택하세요"
    )
    
    # 폴더가 선택되지 않은 경우 종료
    if not selected_folder:
        print("폴더가 선택되지 않았습니다.")
        return
    
    print(f"선택된 폴더: {selected_folder}")
    print("1KB 미만의 파일 삭제를 시작합니다...")
    
    # 작은 파일 삭제 실행
    delete_small_files(selected_folder)
    
    print("작업 완료!")
    
    # GUI 종료
    root.destroy()

if __name__ == "__main__":
    main()