# 랜덤 제너레이터 추가
# 지영님 코드 추가
# 병규님 코드 추가
# 재호님 코드 수정
from mysql import connector
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import plotly.express as px
import matplotlib.pyplot as plt
import random
from random import randrange
import plotly.graph_objects as go
# from sqlalchemy import create_engine



# 페이지 전환 부분


def main():
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "p_main"
    if st.session_state["current_page"] == "p_main":
        p_main()
    elif st.session_state["current_page"] == 'get_marketlist':
        get_marketlist()
    elif st.session_state["current_page"] == 'category_info':
        category_info()
    elif st.session_state["current_page"] == 'registration':
        registration()
    elif st.session_state["current_page"] == 'Customer_dashboard':
        Customer_dashboard()
    elif st.session_state["current_page"] == 'manage':
        manage_count()
    elif st.session_state["current_page"] == 'sales_over_time':
        sales_over_time()
    elif st.session_state["current_page"] == 'age_group_sales':
        age_group_sales()
    elif st.session_state["current_page"] == 'yeary_sales_by_seasons':
        yeary_sales_by_seasons()
    elif st.session_state["current_page"] == 'Time_dashboard':
        Time_dashboard()
    elif st.session_state["current_page"] == 'sale_gendertor':
        sale_gendertor()
    elif st.session_state["current_page"] == 'yeary_sales_by_month':
        yeary_sales_by_month()
    elif st.session_state["current_page"] == 'sales':
        sales()
    elif st.session_state["current_page"] == 'random_his_generator':
        random_his_generator()
    elif st.session_state["current_page"] == 'AD_email':
        AD_email()
    elif st.session_state["current_page"] == 'get_month_by_visit':
        get_month_by_visit()


def conn_db(host, port, user, pw, database):
    conn = connector.connect(host=host, user=user,
                             password=pw, port=port, database=database)
    return conn


def p_main():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.title("고객관련")
        if st.button('회원가입'):
            st.session_state["current_page"] = "registration"

    with col2:
        st.title("상품 관리")
        if st.button('제품 카테고리'):
            st.session_state["current_page"] = "category_info"
        if st.button('재고관리'):
            st.session_state["current_page"] = "manage"
        if st.button('신상품 등록'):
            st.session_state["current_page"] = "get_marketlist"
        if st.button('주문내역 생성'):
            st.session_state["current_page"] = "random_his_generator"

    with col3:
        st.title("관리")
        # if st.button('주문'):
        #     st.session_state["current_page"] = "order"
        if st.button('시간 기준 분석'):
            st.session_state["current_page"] = "Time_dashboard"
        if st.button('고객 정보 기준 분석'):
            st.session_state["current_page"] = "Customer_dashboard"
        if st.button('매출관리'):
            st.session_state["current_page"] = "sales"
        if st.button('AD_email'):
            st.session_state["current_page"] = "AD_email"
        if st.button('get_month_by_visit'):
            st.session_state["current_page"] = "get_month_by_visit"


def AD_email():
    # st 타이틀설정
    st.title("사이트방문 도메인 빈도수 파악")
    # 외부 함수 conn_db불러와서 디비 연결
    connection = conn_db(host=HOST, port=PORT, user=USER,
                         pw=PW, database=DATABASE)
    # df_orderHis, df_customers 필요한 table 블러오기
    df_customers = pd.DataFrame(pd.read_sql(
        "SELECT * FROM customers", connection))
    df_orderHis = pd.DataFrame(pd.read_sql(
        "SELECT * FROM order_his", connection))

    # merge함수 사용하여 두 테이블 연결
    df_sale_gender = pd.merge(df_orderHis, df_customers, on='customer_id')
    # order_date colums의 to_datetime
    df_sale_gender['order_date'] = pd.to_datetime(df_sale_gender['order_date'])

    # 날짜의 최소값 최대값으로 기간 선정
    first_purchase_date = pd.to_datetime(df_sale_gender['order_date'].min())
    last_purchase_date = pd.to_datetime(df_sale_gender['order_date'].max())
    selected_range = st.slider(
        '분석 범위를 설정해 주세요',
        min_value=first_purchase_date.date(),
        max_value=last_purchase_date.date(),
        value=(first_purchase_date.date(), last_purchase_date.date()),
        step=timedelta(days=1)
    )
    first_purchase_date = selected_range[0]
    last_purchase_date = selected_range[1]
    # 기간 필터
    df_sale_gender = df_sale_gender.query(
        '@first_purchase_date <= order_date <= @last_purchase_date')

    df_sale_gender['email_domain'] = df_sale_gender['email'].str.split(
        '@').str[1]
    visits_domain = df_sale_gender.groupby(
        'email_domain')['visit_count'].sum().reset_index()

    # Plot the data using Plotly Express
    fig = px.bar(visits_domain, x='email_domain', y='visit_count', color='email_domain',
                 title='Number of Visits by Email Domain', width=800, height=500)

    # 그래프의 레이아웃을 수정
    fig.update_layout(xaxis_title='이메일 도메인', yaxis_title='총방문수', title_x=0.5)

    # st구현
    st.plotly_chart(fig)

    # 페이지 연결 구동
    if st.button("메인페이지로..."):
        connection.commit()
        connection.close()
        st.session_state["current_page"] = "p_main"

# 3


def get_month_by_visit():
    def get_month(date):
        return date.month
    # st 타이틀설정
    st.title("월별 방문자 조회")
    # 외부 함수 conn_db불러와서 디비 연결
    connection = conn_db(host=HOST, port=PORT, user=USER,
                         pw=PW, database=DATABASE)
    # df_orderHis, df_customers 필요한 table 블로오기
    df_orderHis = pd.DataFrame(pd.read_sql(
        "SELECT * FROM order_his", connection))
    df_customers = pd.DataFrame(pd.read_sql(
        "SELECT * FROM customers", connection))


# Retrieve customer and order history data from the database
    cursor = connection.cursor()
    query = """
        SELECT customers.customer_id, customers.visit_count, order_his.order_date 
        FROM customers 
        INNER JOIN order_his 
        ON customers.customer_id = order_his.customer_id 
        WHERE order_his.order_date BETWEEN '2022-01-01' AND '2022-12-31'
    """
    cursor.execute(query)
    data = cursor.fetchall()
# Close the database connection
    cursor.close()
    connection.close()
# Get the customer ID from the user
    customer_id = st.text_input("고객아이디를 작성해주세요(ID): ")
# Count the number of visits for the customer by month
    visits_by_month = {}
    for row in data:
        if row[0] == customer_id:
            order_date = row[2]
            month = order_date.month
            visit_count = row[1]
            if month not in visits_by_month:
                visits_by_month[month] = visit_count
            else:
                visits_by_month[month] += visit_count
# Create a bar chart of the results using Plotly
    x = []
    y = []
    for month in range(1, 13):
        x.append(date(2022, month, 1).strftime('%B'))
        y.append(visits_by_month.get(month, 0))
    fig = go.Figure(go.Bar(x=x, y=y))
    fig.update_layout(
        title=f'Customer Visits by Month for {customer_id} in 2022', yaxis_title='Visit Count')

    st.plotly_chart(fig)

    # 페이지 연결 구동
    if st.button("메인페이지로..."):
        st.session_state["current_page"] = "p_main"


#####################################################################
def vip():
    # Define function to get month from date
    def get_month(date):
        return date.month

    # Connect to MySQL database
    cnx = connector.connect(
        host="3.35.138.95",
        user="myname",
        password="1234",
        database="mydb",
        port='53686'
    )

    # Retrieve customer and order history data from the database
    cursor = cnx.cursor()
    query = "SELECT customers.customer_id, customers.visit_count, order_his.order_date FROM customers INNER JOIN order_his ON customers.customer_id = order_his.customer_id WHERE order_his.order_date BETWEEN '2022-01-01' AND '2022-12-31'"
    cursor.execute(query)
    data = cursor.fetchall()

    # Close the database connection
    cursor.close()
    cnx.close()

    # Get the customer ID from the user
    customer_id = st.text_input("Enter customer ID: ")

    # Count the number of visits for the customer by month
    visits_by_month = {}
    for row in data:
        if row[0] == customer_id:
            order_date = row[2]
            month = order_date.month
            visit_count = row[1]
            if month not in visits_by_month:
                visits_by_month[month] = visit_count
            else:
                visits_by_month[month] += visit_count

    # Create a bar chart of the results using Plotly
    x = []
    y = []
    for month in range(1, 13):
        x.append(datetime.date(2022, month, 1).strftime('%B'))

        y.append(visits_by_month.get(month, 0))

    fig = go.Figure(go.Bar(x=x, y=y))
    fig.update_layout(
        title=f'Customer Visits by Month for {customer_id} in 2022', yaxis_title='Visit Count')
    st.pyplot(fig)


def sales():  # 전지영님
    conn = connector.connect(host=HOST, user=USER,
                             password=PW, port=PORT, database=DATABASE)
    ### train .csv에 있는 정보 분석 ###
    data_df = pd.DataFrame(pd.read_sql("SELECT * FROM train", conn))
    df = data_df.drop(['Postal Code'], axis=1)

    ############# 다음 데이터를 위한 데이터 재정제 ###############
    df = data_df.drop(['Postal Code'], axis=1)
    ###################### 가장 충성도가 높은 고객정보 PLOT 출력 ###########################
    best_customer = df.pivot_table(
        values="Sales", index="Customer Name", aggfunc="sum")
    best_customer = best_customer.sort_values(
        by=['Sales'], ascending=False).head(20)
    # we will have to reset the index to add the customer name into dataframe
    best_customer.reset_index(inplace=True)
    best_customer['Sales'] = best_customer['Sales'].round(2)
    fig = plt.figure(figsize=(15, 10))
    # creating the bar plot
    plt.bar(best_customer['Customer Name'], best_customer['Sales'], color='maroon',
            width=0.4)
    plt.xlabel("Best Customers", fontsize=15)
    plt.ylabel("Revenue", fontsize=15)
    plt.xticks(fontsize=12, rotation=90)
    plt.yticks(fontsize=12)
    for i, v in enumerate(best_customer['Sales']):
        plt.text(i, v/2, str(v), ha='center',
                 va='center', rotation=90, color='white')
    st.title('가장 충성도가 높은 고객정보 PLOT 출력')
    st.pyplot(fig)
    # 지역별 판매량
    ###################### 가장 판매량이 높은 지역 PLOT 출력 ###########################
    best_cities = df.pivot_table(values="Sales", index="City", aggfunc="sum")
    best_cities = best_cities.sort_values(
        by=['Sales'], ascending=False).head(20)
    best_cities.reset_index(inplace=True)
    best_cities['Sales'] = best_cities['Sales'].round(2)
    fig = plt.figure(figsize=(15, 10))

    # best_cities 그래프
    fig = plt.figure(figsize=(15, 10))
    plt.bar(best_cities['City'], best_cities['Sales'],
            color='maroon', width=0.4)
    plt.xlabel("Best Cities", fontsize=15)
    plt.ylabel("Revenue", fontsize=15)
    plt.xticks(fontsize=12, rotation=90)
    plt.yticks(fontsize=12)
    for i, v in enumerate(best_cities['Sales']):
        plt.text(i, v/2, str(v), ha='center', va='center',
                 rotation=90, color='white', fontsize=8)
    st.title('가장 판매량이 높은 지역 PLOT 출력')
    st.pyplot(fig)
    # 이름으로 그룹핑하여 평균값 구하기
    locations = best_cities.groupby('Sales').mean()
    from sklearn.cluster import KMeans
    ###################### 카테고리별 판매량  PLOT 출력 ###########################
    best_category = df.pivot_table(
        values="Sales", index="Category", aggfunc="sum")
    best_category = best_category.sort_values(
        by=['Sales'], ascending=False).head()
    best_category.reset_index(inplace=True)
    best_category['Sales'] = best_category['Sales'].round(2)
    fig = plt.figure()
    plt.pie(best_category['Sales'],
            labels=best_category['Category'], autopct='%1.1f%%')
    plt.axis('equal')
    st.title('카테고리별 판매량  PLOT 출력')
    st.pyplot(fig)
    if st.button("메인페이지로..."):
        st.session_state["current_page"] = "p_main"


def sale_gendertor():  # 손병규님
    # st 타이틀설정
    st.title("제품 및 성별에 따른 총 매출액")
    # 외부 함수 conn_db불러와서 디비 연결
    connection = conn_db(host=HOST, port=PORT, user=USER,
                         pw=PW, database=DATABASE)
    # df_orderHis, df_customers 필요한 table 블로오기
    df_orderHis = pd.DataFrame(pd.read_sql(
        "SELECT * FROM order_his", connection))
    df_customers = pd.DataFrame(pd.read_sql(
        "SELECT * FROM customers", connection))

    # merge함수 사용하여 두 테이블 연결
    df_sale_gender = pd.merge(df_orderHis, df_customers, on='customer_id')
    ######################################################################################
    # order_date colums의 to_datetime
    df_sale_gender['order_date'] = pd.to_datetime(df_sale_gender['order_date'])

    # 날짜의 최소값 최대값으로 기간 선정
    first_purchase_date = pd.to_datetime(df_sale_gender['order_date'].min())
    last_purchase_date = pd.to_datetime(df_sale_gender['order_date'].max())
    selected_range = st.slider(
        '분석 범위를 설정해 주세요',
        min_value=first_purchase_date.date(),
        max_value=last_purchase_date.date(),
        value=(first_purchase_date.date(), last_purchase_date.date()),
        step=timedelta(days=1)
    )
    first_purchase_date = selected_range[0]
    last_purchase_date = selected_range[1]
    # 기간 필터
    df_sale_gender = df_sale_gender.query(
        '@first_purchase_date <= order_date <= @last_purchase_date')

    ##############################################################################################

    # 제품과 성별로 가로열을 설정이후 각 그룹의 판매량의 열합계를 계산하여 프레임 생성
    grouped_df = df_sale_gender.groupby(['product_id', 'sex']).agg(
        {'total_amount': 'sum'}).reset_index()

    # 그래프 시각화
    fig = px.line(grouped_df, x='product_id', y='total_amount', color='sex',
                  line_group='sex', color_discrete_map={'M': 'blue', 'F': 'red'}, hover_data={'total_amount': ':.2f'}, width=800, height=500)

    # 마우스를 호버링할 때 나타나는 툴팁의 내용을 설정
    fig.update_traces(
        hovertemplate='<b>Product ID: %{x}</b><br>' + 'Total Amount: $%{y:,.2f}')

    # 그래프의 레이아웃을 수정
    fig.update_layout(xaxis_title='제품ID', yaxis_title='총 매출액', title_x=0.5)

    # st구현
    st.plotly_chart(fig)

    # 페이지 연결 구동
    if st.button("메인페이지로..."):
        connection.commit()
        connection.close()
        st.session_state["current_page"] = "p_main"


def Time_dashboard():
    st.title("시간을 기준으로 분석")
    if st.button("1 : 선택한 기간동안의 제품별 판매 실적"):
        st.session_state["current_page"] = "sales_over_time"
    if st.button("2 : 2022년 월매출 분석"):
        st.session_state["current_page"] = "yeary_sales_by_month"
    if st.button("3 : 시즌별 매출 분석"):
        st.session_state["current_page"] = "yeary_sales_by_seasons"
    if st.button("메인페이지로..."):
        st.session_state["current_page"] = "p_main"


def Customer_dashboard():
    st.title("고객 정보를 기준으로 분석")
    if st.button("1 : 연령대별 선호하는 품목"):
        st.session_state["current_page"] = "age_group_sales"
    if st.button('2 : 성별에 따른 매출 변화'):
        st.session_state["current_page"] = "sale_gendertor"
    if st.button("메인페이지로..."):
        st.session_state["current_page"] = "p_main"


def yeary_sales_by_seasons():  # 조재호님
    st.title("시즌별 분석")
    conn = connector.connect(host=HOST, user=USER,
                             password=PW, port=PORT, database=DATABASE)
    product_id = st.number_input("Enter a product ID: ", step=1)
    query = f"SELECT * FROM order_his WHERE product_id = {product_id};"
    cursor = conn.cursor()
    cursor.execute(query)
    order_history = pd.DataFrame(cursor.fetchall())

    # order_his테이블에서 유저가 입력한 product_id를 조회해서 데이터 받아오기
    query = f"SELECT * FROM order_his WHERE product_id = {product_id};"

    # order_his테이블 판다스 데이터프레임형식으로 로드하기
    order_history = pd.read_sql(query, conn)

    # order_date칼럼 datetime형식으로 변환
    order_history['order_date'] = pd.to_datetime(order_history['order_date'])

    # 시즌을 월별로 정의
    seasons = {
        'Winter': [12, 1, 2],
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Fall': [9, 10, 11]
    }

    # 오더별로 시즌정리(뭐하는 코드인지 사실 잘 모름... 해석가능한사람은 주석달아주세요.)
    edges = [0, 3, 6, 9, 12]
    order_history['season'] = pd.cut(
        order_history['order_date'].dt.month,
        bins=edges,
        labels=list(seasons.keys()),
        include_lowest=True
    )

    # 필터된 테이블을 시즌별 주문량으로 엮기
    quantity_by_season = order_history.groupby('season')['quantity'].sum()

    # 시즌별 총 금액 계산
    total_amount_by_season = order_history.groupby('season')[
        'total_amount'].sum()

    # x좌표 순서대로 정의하기
    season_categories = ['Spring', 'Summer', 'Fall', 'Winter']

    # 시즌순서대로 칼럼 나열하기
    order_history['season'] = pd.Categorical(
        order_history['season'], categories=season_categories, ordered=True)

    # 시즌별 총주문금액 그래프 만들기
    fig = px.bar(
        x=quantity_by_season.index,
        y=quantity_by_season.values,
        color=total_amount_by_season.index,
        text=[
            f"Total Amount: ₩{total:,}" for total in total_amount_by_season.values],
        labels={'x': 'Season', 'y': 'Total Amount'},
        title='Total Amount by Season'
    )
    # 콤마와 원화기호 설정
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    # y축 포맷 설정
    fig.update_yaxes(
        tickprefix='',
        tickformat=',',
    )
    st.plotly_chart(fig)

    if st.button("메인페이지로..."):
        st.session_state["current_page"] = "p_main"


def yeary_sales_by_month():  # 조재호님
    st.title("22년 월별 매출 분석")
    conn = connector.connect(host=HOST, user=USER,
                             password=PW, port=PORT, database=DATABASE)
    df = pd.DataFrame(pd.read_sql("SELECT * FROM order_his", conn))
    df.to_csv('order_his.csv', index=False)
    # 애매한 order_his보다 order_history라는 이름으로 변수저장
    order_history = pd.read_csv('order_his.csv')
    # order_history.isnull().sum() 없는거 알지면 혹시모르는 결측치 처리를 위한 엑스트라 스텝
    # order_history.csv 만드는과정으로 똑같이 market_list.csv만듬
    df = pd.read_sql("SELECT * FROM market_list", conn)
    df.to_csv('market_list.csv', index=False)
    market_list = pd.read_csv('market_list.csv')
    # market_list랑 order_history 데이터프레임을 합치고 싶었으나 서로 id랑 product_id가 달라서 마켓리스트 칼럼하나 손봄
    market_list = market_list.rename(columns={'id': 'product_id'})
    # product_id를 바탕으로 두 데이터프레임 결합. 이렇게 하면 product_id를 이용해 product_name을 참조할수 있음
    merged_df = pd.merge(order_history, market_list, on='product_id')
    # 합쳐진 df에 order_date을 판다스가 이해할수있는 to_datetime으로 변환하는 과정
    merged_df['order_date'] = pd.to_datetime(merged_df['order_date'])
    # Define month order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    # Convert month column to Categorical data type with the specified order
    merged_df['month'] = pd.Categorical(merged_df['order_date'].dt.strftime(
        '%B'), categories=month_order, ordered=True)
    # Filter for 2022 sales
    sales_2022 = merged_df[merged_df['order_date'].dt.year == 2022]
    # Group by month and sum total amount
    monthly_sales = sales_2022.groupby('month')['total_amount'].sum()
    # Create bar chart with sorted month order
    fig = plt.figure(figsize=(15, 6))
    plt.bar(month_order, monthly_sales)
    plt.title('Monthly Sales')
    plt.xlabel('Month')
    plt.ylabel('Total Amount')
    st.pyplot(fig)

    if st.button("메인페이지로..."):
        st.session_state["current_page"] = "p_main"


def age_group_sales():  # 손준성님 1
    st.title("세대별 구매 성향 분석 ")
    conn = connector.connect(host=HOST, user=USER,
                             password=PW, port=PORT, database=DATABASE)
    data_df = pd.DataFrame(pd.read_sql("SELECT * FROM order_his", conn))
    customer_df = pd.DataFrame(pd.read_sql("SELECT * FROM customers", conn))
    merge_df = pd.merge(data_df, customer_df, on='customer_id')
    merge_df1 = merge_df.copy()

    # 연령대 구하기
    today = date.today().year
    merge_df1['date_birth'] = pd.to_datetime(merge_df1['date_birth'])
    merge_df1['Age'] = today - merge_df1['date_birth'].dt.year

    def age(age):
        if age < 14:
            return 'child'
        if age < 20 and age >= 14:
            return 'teenagers'
        if age < 30 and age >= 20:
            return '20th'
        if age < 40 and age >= 30:
            return '30th'
        if age < 50 and age >= 40:
            return '40th'
        if age < 60 and age >= 50:
            return '50th'
        if age < 70 and age >= 60:
            return '60th'
        if age < 80 and age >= 70:
            return '70th'
        if age < 90 and age >= 80:
            return '80th'
        else:
            return 'elder'

    merge_df1['group'] = merge_df1['Age'].apply(age)

    # 그래프 그리기
    group_cat = merge_df1.groupby(['group', 'category_id'])['quantity'].sum()
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    group_cat.unstack().plot(kind='bar', stacked=True, ax=ax1)
    ax1.set_xlabel('group')
    ax1.set_ylabel('Quantity')
    ax1.set_title('Quantity by Group and Category ID')
    plt.xticks(rotation=0)
    st.pyplot(fig1)

    product_id = st.number_input("분석하실 product_id를 적어주세요", step=1)
    filtered_data = merge_df1[merge_df1['product_id'] == product_id]
    sales_by_age = filtered_data.groupby('group')['quantity'].sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(sales_by_age.index, sales_by_age.values)
    ax.set_xlabel('group')
    ax.set_ylabel('Quantity')
    ax.set_title('Quantity by Group and Product ID')
    plt.xticks(rotation=0)
    st.pyplot(fig)
    # MySQL 연결 종료
    conn.close()

    if st.button("메인페이지로..."):
        st.session_state["current_page"] = "p_main"


def manage_count():

    col1, col2 = st.columns(2)
    with col1:
        # 마켓리스트 테이블에서 재고량 num 미만일경우 알림띄어줌#조재호님 코드부분
        num = st.number_input("몇개미만 남아있는 제품을 알고 싶으세요?", step=1)
        query = f"SELECT id, product_name, product_count FROM market_list WHERE product_count < {num}"
        connection_result = conn_db(
            host=HOST, port=PORT, user=USER, pw=PW, database=DATABASE)
        cursor = connection_result.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        # 재고량 num미만일 제품이 있을경우 메세지출력
        if len(results) > 0:
            for row in results:
                st.write(f"제품명 : {row[1]} (ID {row[0]}) 이(가) {row[2]}개 남았습니다.")

    with col2:
        # 제품 아이디로 검색해서 재고 확인
        num_2 = st.number_input("어떤 제품의 재고가 알고 싶으신가요", step=1)
        query_2 = f"select product_name,product_count FROM market_list where id = {num_2}"
        connection_result = conn_db(
            host=HOST, port=PORT, user=USER, pw=PW, database=DATABASE)
        cursor = connection_result.cursor()
        cursor.execute(query_2)
        results_2 = cursor.fetchall()
        if len(results_2) == 0:
            st.write("해당 제품이 없습니다.")
        else:
            st.write(
                f"ID: {num_2}, 제품명: {results_2[0][0]}은(는) {results_2[0][1]}개 남았습니다.")
    st.title('재고 추가하기')
    p_id = st.number_input('주문하고자 하는 제품 아이디를 입력해주세요', step=1)
    add_num = st.number_input("원하는 수량을 입력해주세요", step=1)
    if st.button("완료"):
        category_query = f"SELECT category_id FROM market_list WHERE id = {p_id}"
        price_query = f"SELECT product_price FROM market_list WHERE id={p_id}"
        cursor.execute(category_query)
        category_id = cursor.fetchone()[0]
        cursor.execute(price_query)
        price = int(cursor.fetchone()[0])
        current_time = datetime.now()
        order_date = current_time.strftime("%Y-%m-%d")
        total_price = add_num * price
        st.write("재고가 추가되었습니다.")

        sql = f"UPDATE market_list SET product_count = product_count + {add_num} WHERE id = {p_id}"
        sql_2 = f"INSERT INTO market_order (category_id, product_id, order_date, quantity, total_price) VALUES ({category_id}, {p_id}, '{order_date}', {add_num}, {total_price})"
        cursor.execute(sql)
        cursor.execute(sql_2)
        connection_result.commit()
    if st.button("메인페이지로..."):
        connection_result.commit()
        connection_result.close()
        st.session_state["current_page"] = "p_main"


def sales_over_time():
    connection_result = conn_db(
        host=HOST, port=PORT, user=USER, pw=PW, database=DATABASE)
    col1, col2 = st.columns(2)
    with col1:
        p_id = st.number_input("제품아이디 검색", step=1)
        if p_id:
            try:
                sql_2 = f"select * from market_list where id = '{p_id}'"
                cursor = connection_result.cursor()
                cursor.execute(sql_2)
                result = cursor.fetchall()

                st.write('카테고리 :', result[0][1])
                st.write('제품이름 :', result[0][2])
                st.write('가격 :', result[0][3])
                st.write('설명 :', result[0][4])
                st.write('재고 :', result[0][5])
            except:
                st.write('제품아이디가 등록되어있지 않습니다.')
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 4, 10)
    with col2:
        st.write('가장 많이 팔리상품 TOP 3')
        # product_id, quantity합을 total_sales로 정의하여 상위 3개의 제일 잘팔린 아이템을 출력해주는 쿼리문
        query = f"SELECT product_id, SUM(quantity) AS total_sales FROM order_his WHERE order_date >= '{start_date}' AND order_date <= '{end_date}' GROUP BY product_id ORDER BY total_sales DESC LIMIT 3"

        # 판다를 이용하여 해당 쿼리문을 데이터프레임형식으로 저장
        df = pd.read_sql(query, connection_result)
        st.write(df)
    # 카테고리 선택
    selected_ca = st.checkbox("ALL")
    if selected_ca:
        selected_id = "category_id"
    else:
        selected_id_list = []
        if st.checkbox("1 : FOOD"):
            selected_id_list.append(1)
        if st.checkbox("2 : APPLIANCES"):
            selected_id_list.append(2)
        if st.checkbox("3 : PET"):
            selected_id_list.append(3)
        if st.checkbox("4 : ELECTRONIC"):
            selected_id_list.append(4)
        if st.checkbox("5 : FASHION"):
            selected_id_list.append(5)
        if st.checkbox("6 : HOME"):
            selected_id_list.append(6)
        if len(selected_id_list) == 0:
            st.warning("하나 이상의 카테고리를 선택하세요")
            return
        selected_id = ','.join(map(str, selected_id_list))

    # 데이터프레임 불러오기
    sql = f"SELECT * FROM order_his WHERE category_id in ({selected_id})"

    df = pd.read_sql(sql, connection_result)
    st.title("선택한 기간동안의 제품별 판매 실적")

    selected_range = st.slider(
        '분석 범위를 설정해 주세요',
        min_value=start_date,
        max_value=end_date,
        value=(start_date, end_date),
        step=timedelta(days=1)
    )
    start_date = selected_range[0].date()
    end_date = selected_range[1].date()
    # 선택한 기간의 데이터 필터링
    filtered_df = df[(df['order_date'] >= start_date)
                     & (df['order_date'] <= end_date)]
    # 제품별 판매 실적을 막대 그래프로 시각화
    sales_by_product = filtered_df.groupby(
        'product_id')['quantity'].sum().reset_index()
    fig = px.bar(sales_by_product, x='product_id',
                 y='quantity', color='quantity')
    st.plotly_chart(fig)

    if st.button("메인페이지로..."):
        connection_result.commit()
        connection_result.close()
        st.session_state["current_page"] = "p_main"


def category_info():
    sql = "SELECT * FROM category"
    connection_result = conn_db(
        host=HOST, port=PORT, user=USER, pw=PW, database=DATABASE)
    cursor = connection_result.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    for i in range(len(result)):
        st.write(f"카테고리 ID : {result[i][0]}")
        st.write(f"카테고리 명 : {result[i][1]}")

    connection_result.close()
    if st.button("메인페이지로..."):
        st.session_state["current_page"] = "p_main"


def get_marketlist():
    st.write('관리자만 사용해주세요')
    category_id = st.number_input('카테고리 id', step=1)
    product_name = st.text_input("제품이름")
    product_price = st.number_input('제품 가격', step=1)
    product_details = st.text_input('제품 설명')
    product_count = st.number_input('제품 수량', step=1)
    sql = f"insert into maket_list (category_id, product_name,product_price ,product_details, product_count) values({category_id},'{product_name}',{product_price},'{product_details}',{product_count})"
    if st.button("완료."):
        # db연결
        connection_result = conn_db(
            host=HOST, port=PORT, user=USER, pw=PW, database=DATABASE)
        cursor = connection_result.cursor()
        cursor.execute(sql)
        connection_result.commit()
        connection_result.close()
        st.session_state["current_page"] = "p_main"
    if st.button("메인페이지로..."):
        st.session_state["current_page"] = "p_main"


def registration():
    st.title('회원가입')
    st.write('관리자만 사용하시오')
    min_date = datetime(1800, 1, 1)
    max_date = datetime(2023, 12, 31)

    customer_id = st.text_input('id를 입력해주세요')
    customer_password = st.text_input('비밀번호를 등록해주세요')
    first_name = st.text_input('이름')
    last_name = st.text_input('성')
    e_mail = st.text_input("이메일")
    date_birth = st.date_input('생년월일', min_value=min_date, max_value=max_date)
    sex = st.radio('성별', ('M', 'F'))
    address = st.text_input('주소')
    phone_num = st.text_input('전화번호')
    sql = f"insert into customers (customer_id, customer_password, first_name, last_name,email,date_birth, address,phone_num,sex) values ('{customer_id}','{customer_password}','{first_name}','{last_name}','{e_mail}','{date_birth}','{address}','{phone_num}','{sex}')"
    if st.button("완료."):
        # db연결
        connection_result = conn_db(
            host=HOST, port=PORT, user=USER, pw=PW, database=DATABASE)
        cursor = connection_result.cursor()
        cursor.execute(sql)
        connection_result.commit()
        connection_result.close()
        st.session_state["current_page"] = "p_main"
    if st.button("메인페이지로..."):
        st.session_state["current_page"] = "p_main"


def random_his_generator():
    # 아이템 100개 채워넣기
    conn = conn_db(host=HOST, port=PORT, user=USER, pw=PW, database=DATABASE)
    st.title('주문내역을 랜덤으로 생성')
    num = st.number_input('몇개의 주문내역을 생성하시겠습니까? :', step=1)
    if st.button('실행'):
        for i in range(num):
            # customers 테이블에서 랜덤 고객 추출
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM customers ORDER BY RAND() LIMIT 1")
            random_customer = cursor.fetchone()
            random_customer_id = random_customer[0]
            print(random_customer)

            # market_list에서 재고가 1개라도 있는 랜덤 제품 추출
            cursor.execute(
                "SELECT * FROM market_list WHERE product_count > 0 ORDER BY RAND() LIMIT 1")
            random_product = cursor.fetchone()
            random_product_id = random_product[0]
            random_product_category_id = random_product[1]
            random_product_price = random_product[3]
            print(random_product)

            # 랜덤 날짜 생성
            start_date = datetime(2022, 1, 1)
            end_date = datetime(2023, 4, 11)
            random_days = random.randrange((end_date - start_date).days)
            random_date = start_date + timedelta(days=random_days)
            print(random_date.strftime('%Y-%m-%d'))

            # 랜덤 주문 수량 생성 (최대값 10)
            random_quantity = randrange(1, 10)
            print(random_quantity)

            # 총주문액
            total_amount = int(random_product_price) * int(random_quantity)
            print(total_amount)

            # order_his table [order_id, customer_id, category_id, product_id, order_date, quantity, total_amount]

            # 주문내역에 삽입할 것들
            sql = "INSERT INTO order_his (customer_id, category_id, product_id, order_date, quantity, total_amount) VALUES (%s, %s, %s, %s, %s, %s)"
            val = (random_customer_id, random_product_category_id,
                   random_product_id, random_date, random_quantity, total_amount)
            cursor.execute(sql, val)

            # market_list에서 해당 제품의 재고(product_count) 차감
            new_product_count = random_product[5] - random_quantity
            if new_product_count < 0:
                print("Out of stock. Generating a new random order.")
                conn.rollback()
            else:
                sql = "UPDATE market_list SET product_count = %s WHERE id = %s"
                val = (new_product_count, random_product_id)
                cursor.execute(sql, val)

                # 고객테이블 visit_count와 total_amount 갱신
                # increment visit_count by 1
                visit_count = random_customer[9] + 1
                # add total_amount to the current sum
                total_amount_sum = random_customer[10] + total_amount

                sql = "UPDATE customers SET visit_count = %s, total_spend = %s WHERE customer_id = %s"
                val = (visit_count, total_amount_sum, random_customer_id)
                cursor.execute(sql, val)

                conn.commit()
        st.write('내역이 저장되었습니다')
    if st.button("메인페이지로..."):
        st.session_state["current_page"] = "p_main"



if __name__ == "__main__":
    main()