<template>
  <div class="classification">
    <main class="main-content container">
      <div class="container-fluid">
        <div class="page-header d-md-flex justify-content-between align-items-center">
        </div>
        <div class="row">
          <div class="page-header d-md-flex justify-content-between align-items-center">
            <div class="col-lg-6 col-md-6 col-sm-6">
              <h4>Phân tích</h4>
              <nav aria-label="breadcrumb">
                <ol class="breadcrumb">
                </ol>
              </nav>
            </div>
            <div class="col-lg-6 col-md-6 col-sm-6">
              <div class="user-request d-flex align-items-center justify-content-end">
                <div class="dropdown camera d-flex ">
                  <select class="form-select select_cam_option" aria-label="Default select example">
                    <!-- <option selected>Open this select menu</option> -->
                    <option value="Cam1">Cam1</option>
                    <option value="Cam2">Cam2</option>
                  </select>
                </div>
                <div class="dropdown year">
                  <select class="form-select select_year_option" aria-label="Default select example">
                    <option value="2022">2022</option>
                    <option value="2021">2021</option>
                    <option value="2020">2020</option>
                  </select>
                </div>
                <b-button variant="success" style="height: fit-content" @click="getAllAPI()">Request</b-button>
              </div>
            </div>
          </div>
        </div>

        <!-- end::page header -->

        <div class="row">
          <div class="col-md-3">
            <div class="card">
              <div class="card-body">
                <div class="data-vehicle-intro d-flex justify-content-between align-items-center">
                  <div>
                    <h6>Truck</h6>
                    <h4 class="m-b-0 font-weight-bold">
                      {{ data_vehicle_intro[0] }}
                    </h4>
                  </div>
                  <div class="icon-vehicles">
                    <img src="../assets/truck-icon.png" alt="Search" style="width: 70px" class="icon" />
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div class="col-md-3">
            <div class="card">
              <div class="card-body">
                <div class="data-vehicle-intro d-flex justify-content-between align-items-center">
                  <div>
                    <h6>Bus</h6>
                    <h4 class="m-b-0 font-weight-bold">
                      {{ data_vehicle_intro[1] }}
                    </h4>
                  </div>
                  <div class="icon-vehicles">
                    <img src="../assets/bus-icon.png" alt="Search" style="width: 70px" class="icon" />
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div class="col-md-3">
            <div class="card">
              <div class="card-body">
                <div class="data-vehicle-intro d-flex justify-content-between align-items-center">
                  <div>
                    <h6>Car</h6>
                    <h4 class="m-b-0 font-weight-bold">
                      {{ data_vehicle_intro[2] }}
                    </h4>
                  </div>
                  <div class="icon-vehicles">
                    <img src="../assets/car-icon2.jpg" alt="Search" style="width: 70px" class="icon" />
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div class="col-md-3">
            <div class="card">
              <div class="card-body">
                <div class="data-vehicle-intro d-flex justify-content-between align-items-center">
                  <div>
                    <h6>Motorbycle</h6>
                    <h4 class="m-b-0 font-weight-bold">
                      {{ data_vehicle_intro[3] }}
                    </h4>
                  </div>
                  <div class="icon-vehicles">
                    <img src="../assets/motobycle-icon2.png" alt="Search" style="width: 70px" class="icon" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div class="row">
          <div class="card-header bg-transparent">
            <div class="row align-items-center">
              <div class="col">
                <h5 class="h3 text-black mb-0">
                  Tổng số phương tiện qua lại mỗi tháng
                </h5>
              </div>
            </div>
          </div>
        </div>
        <!-- page chart month line -->
        <div class="row">
          <div class="col-xl-8">
            <div class="card bg-default">
              <div class="card-body">
                <div class="chart">
                  <div class="chartjs-size-monitor">
                    <div class="chartjs-size-monitor-expand">
                      <div class=""></div>
                    </div>
                    <div class="chartjs-size-monitor-shrink">
                      <div class=""></div>
                    </div>
                  </div>

                  <canvas id="chart_column_month" width="400" height="200px"></canvas>
                </div>
              </div>
            </div>
          </div>
          <div class="col-xl-4">
            <div class="card">
              <div class="card-header bg-transparent">
                <div class="row align-items-center">
                  <div class="col">
                    <canvas id="chart_column_monthDougnut" width="400" height="430"></canvas>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div class="row">
          <div data-v-6a4aa1a0="" class="col-lg-3 col-md-12 col-sm-12 p-0">
            <div class="row">
              <div class="col-lg-12 col-md-6 col-md-6">
                <div data-v-6a4aa1a0="" class="card">
                  <div data-v-6a4aa1a0="" class="card-header">Max season - season {{ data_season[0].name }}</div>
                  <div data-v-6a4aa1a0="" class="card-body text-center">
                    <h2 data-v-6a4aa1a0="" class="text-danger font-weight-bold">
                      {{ data_season[0].quantity }}
                    </h2>
                    <p data-v-6a4aa1a0="" class="m-b-0">
                      <i class="fa-solid fa-car"></i>
                      {{ data_season[0].percent }} % in that year
                    </p>
                  </div>
                </div>
              </div>
              <div class="col-lg-12 col-md-6 col-md-6">
                <div data-v-6a4aa1a0="" class="card">
                  <div data-v-6a4aa1a0="" class="card-header">Min season - season {{ data_season[1].name }}</div>
                  <div data-v-6a4aa1a0="" class="card-body text-center">
                    <h2 data-v-6a4aa1a0="" class="text-danger font-weight-bold">
                      {{ data_season[1].quantity }}
                    </h2>
                    <p data-v-6a4aa1a0="" class="m-b-0">
                      <i class="fa-solid fa-bus"></i>
                      {{ data_season[1].percent }} % in that year
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div class="col-lg-9 col-md-12 col-sm-12">
            <div class="card bg-default">
              <div class="card-body">
                <div class="chart">
                  <div class="chartjs-size-monitor">
                    <div class="chartjs-size-monitor-expand">
                      <div class=""></div>
                    </div>
                    <div class="chartjs-size-monitor-shrink">
                      <div class=""></div>
                    </div>
                  </div>
                  <canvas id="chart_column_monthLine" width="400" height="150"></canvas>
                </div>
              </div>
            </div>
          </div>
        </div>
        <!-- doing here -->
        <div class="row">
          <div class="col-lg-9 col-md-12 col-sm-12">
            <div class="card bg-default d-flex justify-center align-center">
              <div class="card-body">
                <div class="chart d-flex" style="justify-content: center">
                  <div class="chartjs-size-monitor">
                    <div class="chartjs-size-monitor-expand">
                      <div class=""></div>
                    </div>
                    <div class="chartjs-size-monitor-shrink">
                      <div class=""></div>
                    </div>
                  </div>
                  <canvas id="chart_rada_year_hour" width="100" height="100"></canvas>
                </div>
              </div>
            </div>
          </div>
          <div class="col-lg-3 col-md-12 col-sm-12">
            <div class="row">
              <div class="col-lg-12 col-md-6 col-md-6">
                <div data-v-6a4aa1a0="" class="card">
                  <div data-v-6a4aa1a0="" class="card-header">Morning</div>
                  <div data-v-6a4aa1a0="" class="card-body text-center">
                    <h2 data-v-6a4aa1a0="" class="text-danger font-weight-bold">
                      {{ data_time_in_day[0].quantity }}
                    </h2>
                    <p data-v-6a4aa1a0="" class="m-b-0">{{ data_time_in_day[0].percent }} % in that day</p>
                  </div>
                </div>
              </div>
              <div class="col-lg-12 col-md-6 col-md-6">
                <div data-v-6a4aa1a0="" class="card">
                  <div data-v-6a4aa1a0="" class="card-header">Afternoon</div>
                  <div data-v-6a4aa1a0="" class="card-body text-center">
                    <h2 data-v-6a4aa1a0="" class="text-danger font-weight-bold">
                      {{ data_time_in_day[1].quantity }}
                    </h2>
                    <p data-v-6a4aa1a0="" class="m-b-0">{{ data_time_in_day[1].percent }} % in that day</p>
                  </div>
                </div>
              </div>
              <div class="col-lg-12 col-md-6 col-md-6">
                <div data-v-6a4aa1a0="" class="card">
                  <div data-v-6a4aa1a0="" class="card-header">Night</div>
                  <div data-v-6a4aa1a0="" class="card-body text-center">
                    <h2 data-v-6a4aa1a0="" class="text-danger font-weight-bold">
                      {{ data_time_in_day[2].quantity }}
                    </h2>
                    <p data-v-6a4aa1a0="" class="m-b-0">{{ data_time_in_day[2].percent }} % in that day</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <!-- de o day -->
        <div data-v-6a4aa1a0="" class="row">
          <div data-v-6a4aa1a0="" class="col">
            <div data-v-6a4aa1a0="" class="card bg-default">
              <div data-v-6a4aa1a0="" class="card-body">
                <div data-v-6a4aa1a0="" class="chart">
                  <div class="chartjs-size-monitor">
                    <div class="chartjs-size-monitor-expand">
                      <div class=""></div>
                    </div>
                    <div class="chartjs-size-monitor-shrink">
                      <div class=""></div>
                    </div>
                  </div>
                  <div data-v-6a4aa1a0="" class="chartjs-size-monitor">
                    <div data-v-6a4aa1a0="" class="chartjs-size-monitor-expand">
                      <div data-v-6a4aa1a0=""></div>
                    </div>
                    <div data-v-6a4aa1a0="" class="chartjs-size-monitor-shrink">
                      <div data-v-6a4aa1a0=""></div>
                    </div>
                  </div>
                  <canvas data-v-6a4aa1a0="" id="chart_column_monthBarPlus" width="800" height="400"
                    style="display: block; height: 320px; width: 640px" class="chartjs-render-monitor"></canvas>
                </div>
              </div>
            </div>
          </div>
        </div>
        <!-- <div class="row">
          <div class="col">
            <div class="card">
              <div class="card-header">
                <i class="fa fa-download text-muted m-r-5"></i>
                Download your earnings in CSV format
              </div>
              <div class="card-body">
                <p class="text-muted text-center">
                  Open it in a spreadsheet and perform your own calculations,
                  graphing etc. The CSV file contains additional details, such
                  as the buyer location.
                </p>
                <div class="row">
                  <div class="col-md-12 d-flex">
                    <div class="reportrange btn btn-outline-primary btn-block">
                      <i class="ti-calendar m-r-10"></i>
                      <span>May 17, 2022 - June 15, 2022</span>
                      <i class="ti-angle-down m-l-10"></i>
                    </div>
                    <b-button variant="success" onclick="downloadFile">
                      <i class="fa fa-download"></i>
                    </b-button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div> -->
      </div>
    </main>
  </div>
</template>
<script src="https://cdn.jsdelivr.net/npm/chart.js@3.8.0/dist/chart.min.js"></script>
<script
  type="text/javascript"
  src="https://yourweb.com/inc/chart.utils.js"
></script>
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
<script>
import axios from "axios";
export default {
  name: "home",
  type: "bar",
  d: new Date(),
   data: () => ({
    BE: [],
    day: new Date().getDate(),
    month: new Date().getMonth() + 1,
    year: new Date().getFullYear(),
    cam: "Cam1",
    task: [],
    width: 2,
    radius: 10,
    padding: 8,
    lineCap: "round",
    value: [0, 2, 5, 9, 5, 10, 3, 5, 0, 0, 1, 8, 2, 9, 0],
    fill: false,
    type: "trend",
    autoLineWidth: false,
    date: new Date(),
    options: {
      format: "DD/MM/YYYY",
      useCurrent: true,
      scales: {
        y: {
          beginAtZero: true,
        },
      },
    },

    // list_time_analysis_choice: [
    //   {
    //     id: 0,
    //     name: "Month",
    //     active: true,
    //   },
    //   {
    //     id: 1,
    //     name: "Day",
    //     active: false,
    //   },
    //   {
    //     id: 2,
    //     name: "Year",
    //     active: false,
    //   },
    // ],
    chartData: {
      "2017-05-13": 2,
      "2017-05-14": 3,
      "2017-05-15": 4,
    },
    months: {
      January: "01",
      February: "02",
      March: "03",
      April: "04",
      May: "05",
      June: "06",
      July: "07",
      August: "08",
      September: "09",
      October: "10",
      November: "11",
      December: "12",
    },
    days: {
      1: 0,
      2: 0,
      3: 0,
      4: 0,
      5: 0,
      6: 0,
      7: 0,
      8: 0,
      9: 0,
      10: 0,
      11: 0,
      12: 0,
      13: 0,
      14: 0,
      15: 0,
      16: 0,
      17: 0,
      18: 0,
      19: 0,
      20: 0,
      21: 20,
      22: 30,
      23: 50,
      24: 30,
      25: 40,
      26: 70,
      27: 50,
      28: 110,
      29: 0,
      30: 0,
    },
    selected_month: "January",
    years: ["2016", "2017", "2018", "2019", "2020", "2021", "2022"],
    selected_year: "2022",
    cameras: ["Cam1", "Cam2"],
    selected_camera: "Cam1",
    // chart column month
    chart_column_month: "",
    // chart column month dougnut
    chart_column_monthDougnut: "",
    chart_column_monthLine: "",
    chart_column_monthBarPlus: "",
    data_vehicle_intro: [0, 0, 0, 0],
    data_season: [
      {
        name: 0,
         quantity: 0,
         percent: 0
      },
      {
        name: 0,
        quantity: 0,
        percent: 0
      }
    ],
    morning : {},
    night :{},
    afternoon : {},
    data_time_in_day: [
      {
          name: 'Morning',
          quantity: 78,
          percent: 12
      },
      {
          name: 'Afternoon',
          quantity: 158,
          percent: 23
      },
      {
          name: 'Night',
          quantity: 86,
          percent: 62
      },
    ]
  }
  ),
  async created() {
    try {
      console.log("@@@: " + this.BE);
      await axios
        .get(`http://localhost:8000/detail?year=2022`)
        .then((response) => {
          this.BE = response.data;
          console.log("@@@@:wddwdwd :", this.BE);
        });
    } catch (e) {
      console.log("Error: ", e);
    }
    console.log("After: ", this.BE);
    await this.fetchAPIIntro();
    await this.fetchAPI();
    await this.fetchAPIChartDonut();
    await this.fetchAPIChartLine();
    await this.fetchAPIChartPlus();
    await this.fetchAPIChartTimeADay();
    await this.fetchAPIDataseason();
  },
  computed: {
    theme() {
      return this.$vuetify.theme.dark ? "dark" : "light";
    },
  },
  methods: {
    // downloadFile() {
    //   consloe.log("Tuan")
    //   // axios({
    //   //       url: 'http://localhost:8000/demo.pdf', // File URL Goes Here
    //   //       method: 'GET',
    //   //       responseType: 'blob',
    //   //   }).then((res) => {
    //   //         var FILE = window.URL.createObjectURL(new Blob([res.data]));
              
    //   //         var docUrl = document.createElement('x');
    //   //         docUrl.href = FILE;
    //   //         docUrl.setAttribute('download', 'file.pdf');
    //   //         document.body.appendChild(docUrl);
    //   //         docUrl.click();
    //   // });
    // },
    async getAllAPI() {
      try {
        var e = document.querySelector(".select_year_option");
        var year = e.options[e.selectedIndex].text;
        e = document.querySelector(".select_cam_option");
        var cam = e.options[e.selectedIndex].text;
        // console.log("@@@: " + );
        await axios
          .get(`http://localhost:8000/detail?year=${year}&cam=${cam}`)
          .then((response) => {
            this.BE = response.data;
            console.log("@@@@:request :", this.BE);
          });
      } catch (e) {
        console.log("Error: ", e);
      }
      console.log("After: ", this.BE);
      await this.fetchAPIIntro();
      await this.fetchAPI();
      await this.fetchAPIChartDonut();
      await this.fetchAPIChartLine();
      await this.fetchAPIChartPlus();
      await this.fetchAPIChartTimeADay();
      await this.fetchAPIDataseason();
      console.log(
        "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
      );
      console.log(this.BE.year_now);
      console.log("s luong truck: ", this.BE[this.BE.year_now].truck);
    },
    async fetchAPIIntro() {
      console.log("fetchAPI intro");
      this.data_vehicle_intro = [this.BE[this.BE.year_now].truck, this.BE[this.BE.year_now].bus, this.BE[this.BE.year_now].car,this.BE[this.BE.year_now].motorbycle];

    },
    async fetchAPI() {
      console.log("fetchAPI");
      const data_fetch = [];
      console.log(
        "Data: ",
        Object.values(this.BE.data_month).forEach((val) => {
          console.log(val.data.all);
          data_fetch.push(val.data.all);
        })
      );
      this.chart_column_month.data.datasets[0].data = data_fetch;
      // console.log('Hello: ', this.BE.data_month.filter(date_item => date_item.id === '1').map(data_item => data_item))
      this.chart_column_month.update();
    },
    async fetchAPIChartDonut() {
      console.log("@@:","fetchAPIChartDonut");
      let all_truck = 0;
      let all_car = 0;
      let all_bus = 0;
      let all_motorbycle = 0;
      Object.values(this.BE.data_month).forEach((val) => {
        console.log(val.data);
        all_truck = all_truck + val.data.truck;
        all_car = all_car + val.data.car;
        all_bus = all_bus + val.data.bus;
        all_motorbycle = all_motorbycle + val.data.motorbycle;
      });
      let data_fetch = [all_truck, all_bus, all_car, all_motorbycle];
      this.chart_column_monthDougnut.data.datasets[0].data = data_fetch;
      this.chart_column_monthDougnut.update();
      // console.log(Math.round((150 / data_fetch.reduce(function(a, b) { return a + b; }, 0)) * 100))
      let total_data_fetch = data_fetch.reduce(function (a, b) {
        return a + b;
      }, 0);
      data_fetch.forEach((item, index) => {
        data_fetch[index] = Math.round((item / total_data_fetch) * 100);
      });
      console.log("du lieu moi: ", data_fetch);
    },
    async fetchAPIChartLine() {
      console.log("fetchAPIChartLine");
      let data_car = [];
      let data_truck = [];
      let data_motorbycle = [];
      let data_bus = [];
      Object.values(this.BE.data_month).forEach((val) => {
        data_car.push(val.data.car);
        data_bus.push(val.data.bus);
        data_motorbycle.push(val.data.motorbycle);
        data_truck.push(val.data.truck);
      });
      this.chart_column_monthLine.data.datasets[0].data = data_truck;
      this.chart_column_monthLine.data.datasets[1].data = data_bus;
      this.chart_column_monthLine.data.datasets[2].data = data_car;
      this.chart_column_monthLine.data.datasets[3].data = data_motorbycle;
      this.chart_column_monthLine.update();
      // set data for max min season
    },
    async fetchAPIChartPlus() {
      console.log("fetchAPIChartPlus");
            let data_truck = [];
      let data_bus = [];
      let data_car = [];
      let data_motorbycle = [];
      let year_bar_plus = Array.from(this.BE.years)
      for(let i = 0;i<year_bar_plus.length;i++){
              data_truck.push(this.BE[year_bar_plus[i]].truck);
      data_bus.push(this.BE[year_bar_plus[i]].bus);
      data_car.push(this.BE[year_bar_plus[i]].car);
      data_motorbycle.push(this.BE[year_bar_plus[i]].motorbycle);
      }
      this.chart_column_monthBarPlus.data.datasets[0].data = data_truck;
      this.chart_column_monthBarPlus.data.datasets[1].data = data_bus;
      this.chart_column_monthBarPlus.data.datasets[2].data = data_car;
      this.chart_column_monthBarPlus.data.datasets[3].data = data_motorbycle;
      this.chart_column_monthBarPlus.update();
    },
    async fetchAPIChartTimeADay() {
      console.log('@@: fetchAPIChartTimeADay',)
      this.morning = this.BE.morning
      this.afternoon = this.BE.afternoon
      this.night = this.BE.night
      let morning_quantity = 0
      let afternoon_quantity = 0
      let night_quantity = 0
      Object.keys(this.morning).forEach(element => morning_quantity = this.morning[element].count + morning_quantity)
      Object.keys(this.afternoon).forEach(element => afternoon_quantity = this.afternoon[element].count + afternoon_quantity)
      Object.keys(this.night).forEach(element => night_quantity = this.night[element].count + night_quantity)
      this.data_time_in_day[0].quantity = morning_quantity
      this.data_time_in_day[0].percent = Math.round((morning_quantity/(morning_quantity+afternoon_quantity+night_quantity))*100,2)
      this.data_time_in_day[1].quantity = afternoon_quantity
      this.data_time_in_day[1].percent = Math.round((afternoon_quantity/(morning_quantity+afternoon_quantity+night_quantity))*100,2)
      this.data_time_in_day[2].quantity = night_quantity
      this.data_time_in_day[2].percent = Math.round((night_quantity/(morning_quantity+afternoon_quantity+night_quantity))*100,2)
      let morning_truck_basic = []
      for(let i = 0; i<24 ; i++){
        morning_truck_basic.push(NaN)
      }
      var morning_truck = [NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]
      var morning_car = [NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]
      var morning_bus = [NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]
      var morning_motorbike = [NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]
      var afternoon_truck = [NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]
      var afternoon_car = [NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]
      var afternoon_bus = [NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]
      var afternoon_motorbike = [NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]
      var night_truck = [NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]
      var night_car = [NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]
      var night_bus = [NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]
      var night_motorbike = [NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN]
      for (let x of [5,6,7,8,9,10,11,12]) {
        morning_truck[x] = this.morning[x].truck
        morning_car[x] = this.morning[x].car
        morning_bus[x] = this.morning[x].bus
        morning_motorbike[x] = this.morning[x].motorcycle
      }
      this.chart_rada_year_hour.data.datasets[0].data = morning_truck
      this.chart_rada_year_hour.data.datasets[1].data = morning_bus
      this.chart_rada_year_hour.data.datasets[2].data = morning_car
      this.chart_rada_year_hour.data.datasets[3].data = morning_motorbike
      for(let x of [13,14,15,16,17]){
        afternoon_truck[x] = this.afternoon[x].truck
        afternoon_car[x] = this.afternoon[x].car
        afternoon_bus[x] = this.afternoon[x].bus
        afternoon_motorbike[x] = this.afternoon[x].motorcycle
      }
      this.chart_rada_year_hour.data.datasets[4].data = afternoon_truck
      this.chart_rada_year_hour.data.datasets[5].data = afternoon_bus
      this.chart_rada_year_hour.data.datasets[6].data = afternoon_car
      this.chart_rada_year_hour.data.datasets[7].data = afternoon_motorbike
      for(let x of [18,19,20,21,22,23,0,1,2,3,4]){
        night_truck[x] = this.night[x].truck
        night_car[x] = this.night[x].car
        night_bus[x] = this.night[x].bus
        night_motorbike[x] = this.night[x].motorcycle
      }
      this.chart_rada_year_hour.data.datasets[8].data = night_truck
      this.chart_rada_year_hour.data.datasets[9].data = night_bus
      this.chart_rada_year_hour.data.datasets[10].data = night_car
      this.chart_rada_year_hour.data.datasets[11].data = night_motorbike
      this.chart_rada_year_hour.update()
    },
    async fetchAPIDataseason() {
      console.log('fetchAPIDataseason')
      this.data_season[0].name = this.BE.season.max.name
      this.data_season[0].quantity = this.BE.season.max.quantity
      this.data_season[0].percent = this.BE.season.max.percent
      this.data_season[1].name = this.BE.season.min.name
      this.data_season[1].quantity = this.BE.season.min.quantity
      this.data_season[1].percent = this.BE.season.min.percent
    },

    onChange() {
      alert(this.date);
    },
    get_Tasks() {
      axios({
        method: "get",
        url:
          "http://localhost:8000/vehicle?year=" +
          this.year +
          "&day=" +
          this.day +
          "&month=" +
          this.month +
          "&camera=" +
          this.cam,
      }).then((response) => (this.task = response.data));
    },
    getTime() {
      alert(this.selected_month);
      alert(this.selected_year);
      axios({
        method: "get",
        url:
          "http://localhost:8000/vehicle?year=" +
          this.selected_year +
          "&day=" +
          "&month=" +
          this.selected_month.number +
          "&camera=" +
          this.cam,
      }).then((response) => (this.task = response.data));
    },
  },
  mounted() {
    // this.downloadFile();
    const ctx = document.getElementById("chart_column_month").getContext("2d");
    this.chart_column_month = new Chart(ctx, {
      type: "bar",
      data: {
        labels: Object.keys(this.months),
        datasets: [
          {
            label: ["# of Votes", "Tuan"],
            data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            backgroundColor: [
              "rgba(255, 0, 0, 0.2)",
              "rgba(255, 127, 0, 0.2)",
              "rgba(255, 255, 0, 0.2)",
              "rgba(127, 255, 0, 0.2)",
              "rgba(0, 255, 0, 0.2)",
              "rgba(0, 255, 127, 0.2)",
              "rgba(0, 255, 255, 0.2)",
              "rgba(0, 127, 255, 0.2)",
              "rgba(0, 0, 255, 0.2)",
              "rgba(127, 0, 255, 0.2)",
              "rgba(255, 0, 255, 0.2)",
              "rgba(255, 0, 127, 0.2)",
            ],
            hoverBackgroundColor: [
              "rgba(255, 0, 0, 0.5)",
              "rgba(255, 127, 0, 0.5)",
              "rgba(255, 255, 0, 0.5)",
              "rgba(127, 255, 0, 0.5)",
              "rgba(0, 255, 0, 0.5)",
              "rgba(0, 255, 127, 0.5)",
              "rgba(0, 255, 255, 0.5)",
              "rgba(0, 127, 255, 0.5)",
              "rgba(0, 0, 255, 0.5)",
              "rgba(127, 0, 255, 0.5)",
              "rgba(255, 0, 255, 0.5)",
              "rgba(255, 0, 127, 0.5)",
            ],
            borderColor: [
              "rgba(255, 0, 0, 1)",
              "rgba(255, 127, 0, 1)",
              "rgba(255, 255, 0, 1)",
              "rgba(127, 255, 0, 1)",
              "rgba(0, 255, 0, 1)",
              "rgba(0, 255, 127, 1)",
              "rgba(0, 255, 255, 1)",
              "rgba(0, 127, 255, 1)",
              "rgba(0, 0, 255, 1)",
              "rgba(127, 0, 255, 1)",
              "rgba(255, 0, 255, 1)",
              "rgba(255, 0, 127, 1)",
            ],
            borderWidth: 1,
          },
        ],
      },
      options: {
        scales: {
          y: {
            stacked: true,
          },
        },
      },
    });
    const data_lines = {
      labels: Object.keys(this.months),
      datasets: [
        {
          label: "Truck",
          data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
          fill: true,
          backgroundColor: "rgb(239, 71, 111,0.3)",
          borderColor: "rgb(239, 71, 111)",
          tension: 0.1,
          yAxesID: "y",
          // borderColor: 'rgba(255, 0, 127, 1)',
        },
        {
          label: "Car",
          data: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
          fill: true,
          backgroundColor: "rgb(255, 209, 102,0.3)",
          borderColor: "rgb(255, 209, 102)",
          tension: 0.2,
          yAxesID: "y1",
          // borderColor: 'rgba(255, 255, 127, 1)',
        },
        {
          label: "Bus",
          data: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
          fill: true,
          backgroundColor: "rgb(6, 214, 160,0.3)",
          borderColor: "rgb(6, 214, 160)",
          tension: 0.2,
          yAxesID: "y2",
          // borderColor: 'rgba(255, 255, 127, 1)',
        },
        {
          label: "Motorbike",
          data: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
          fill: true,
          backgroundColor: "rgb(17, 138, 178,0.3)",
          borderColor: "rgb(17, 138, 178)",
          tension: 0.2,
          yAxesID: "y3",
          // borderColor: 'rgba(255, 255, 127, 1)',
        },
      ],
    };
    const ctxLine = document
      .getElementById("chart_column_monthLine")
      .getContext("2d");
    this.chart_column_monthLine = new Chart(ctxLine, {
      type: "line",
      data: data_lines,
      scales: {
        y: {
          type: "linear",
          display: true,
          position: "left",
        },
        y1: {
          type: "linear",
          display: true,
          position: "right",
          grid: {
            drawOnChartArea: false, // only want the grid lines for one axis to show up
          },
        },
        y2: {
          type: "linear",
          display: true,
          position: "right",
          grid: {
            drawOnChartArea: false, // only want the grid lines for one axis to show up
          },
        },
        y3: {
          type: "linear",
          display: true,
          position: "right",
          grid: {
            drawOnChartArea: false, // only want the grid lines for one axis to show up
          },
        },
      },
    });
    const ctxDougnut = document
      .getElementById("chart_column_monthDougnut")
      .getContext("2d");
    this.chart_column_monthDougnut = new Chart(ctxDougnut, {
      type: "doughnut",
      data: {
        labels: ["Truck", "Bus", "Car", "Motorbike"],
        datasets: [
          {
            label: "My First Dataset",
            data: [300, 50, 100, 45],
            backgroundColor: [
              "rgb(239, 71, 111,0.4)",
              "rgb(255, 209, 102,0.4)",
              "rgb(6, 214, 160,0.4)",
              "rgb(17, 138, 178,0.4)",
            ],
            hoverBackgroundColor:[
              "rgb(239, 71, 111,0.9)",
              "rgb(255, 209, 102,0.9)",
              "rgb(6, 214, 160,0.9)",
              "rgb(17, 138, 178,0.9)",
            ],
            hoverOffset: 4,
          },
        ],
      },
    });
    // bar plus
    const DATA_COUNT = 7;
    const NUMBER_CFG = { count: DATA_COUNT, min: -100, max: 100 };
    const ctxBarPlus = document
      .getElementById("chart_column_monthBarPlus")
      .getContext("2d");
    this.chart_column_monthBarPlus = new Chart(ctxBarPlus, {
      type: "bar",
      data: {
        labels: [2020, 2021, 2022],
        datasets: [
          {
            label: "Truck",
            type: "bar",
            stack: "Base",
            backgroundColor: "rgb(120, 171, 195,0.8)",
            backgroundColorHover: "rgb(120, 171, 195,0.4)",
            data: [34, 35, 36],
          },
          {
            label: "Bus",
            type: "bar",
            stack: "Base",
            backgroundColor: "rgb(229, 199, 106,0.8)",
            backgroundColorHover: "rgb(229, 199, 106,0.4)",
            data: [ 19, 20, 21],
          },
          {
            label: "Car",
            type: "bar",
            stack: "Base",
            backgroundColor: "rgb(212, 137, 96,0.8)",
            backgroundColorHover: "rgb(212, 137, 96,0.4)",
            data: [24, 25, 26],
          },
          {
            label: "Motorbycle",
            type: "bar",
            stack: "Base",
            backgroundColor: "rgb(115, 150, 114,0.8)",
            backgroundColorHover: "rgb(115, 150, 114,0.4)",
            data: [14, 15, 16],
          },
        ],
      },
      options: {
        scales: {
          xAxes: [
            {
              //stacked: true,
              stacked: true,
              ticks: {
                beginAtZero: true,
                maxRotation: 0,
                minRotation: 0,
              },
            },
          ],
          yAxes: [
            {
              stacked: true,
            },
          ],
        },
      },
    });
    const data_barplus = {
      labels: ["Truck", "Bus", "Car", "Motorbike", "ap", "loa", "askd"],
      datasets: [
        {
          label: "Dataset 1",
          data: [1, 2, 3, 4, 5, 6, 7],
          backgroundColor: "rgb(154, 156, 233)",
          stack: "Stack 0",
        },
        {
          label: "Dataset 2",
          data: [1, 2, 3, 4, 5, 5, 6],
          backgroundColor: "rgb(154, 156, 233)",
          stack: "Stack 0",
        },
        {
          label: "Dataset 3",
          data: [1, 2, 2, 4, 5, 5, 6],
          backgroundColor: "rgb(154, 156, 233)",
          stack: "Stack 1",
        },
      ],
    };
    const labels = [];
    for (let i = 0; i < 24; ++i) {
      labels.push(i.toString());
    }
    const crada = document
      .getElementById("chart_rada_year_hour")
      .getContext("2d");
    this.chart_rada_year_hour = new Chart(crada, {
      type: "line",
      data: {
        labels: labels,
        // datapoints: [0, 20, 20, 60, 60, 120, 15, 180, 120, 125, 105, 110, 170],
        datasets: [
          // 1
          {
            label: "Truck",
            data: [0, 20, 20, 60, 60, 120, NaN, 180, 120, 125, 105, 110, 170,0, 20, 20, 60, 60, 120, NaN, 180, 120, 125, 105, 110, 170],
            borderColor: "rgb(239, 71, 111,0.7)",
            fill: false,
            cubicInterpolationMode: "monotone",
            tension: 0.4,
          },
          // 2
          {
            label: "Car",
            data: [NaN, NaN, NaN, 60, 60, 120, 15, 180, 120, 125, 105, 110, 170],
            borderColor: "rgb(255, 209, 102,0.7)",
            fill: false,
            tension: 0.5,
            cubicInterpolationMode: "monotone",
          },
          // 3
          {
            label: "Bus",
            data: [0, 20, 20, 60, 60, 120, 15, 180, 120, 125, 105, 110, 170],
            borderColor: "rgb(6, 214, 160,0.7)",
            fill: false,
            cubicInterpolationMode: "monotone",
          },
          // 4
          {
            label: "Motorbike",
            data: [0, 20, 20, 60, 60, 120, 15, 180, 120, 125, 105, 110, 170],
            borderColor: "rgb(17, 138, 178,0.7)",
            fill: false,
            cubicInterpolationMode: "monotone",
          },
          // 5
          {
            label: "",
            data: [0, 20, 20, 60, 60, 120, 15, 180, 120, 125, 105, 110, 170],
            borderColor: "rgb(239, 71, 111,0.7)",
            fill: false,
            cubicInterpolationMode: "monotone",
          },
          // 6
          {
            label: "",
            data: [0, 20, 20, 60, 60, 120, 15, 180, 120, 125, 105, 110, 170],
            borderColor: "rgb(255, 209, 102,0.7)",
            fill: false,
            cubicInterpolationMode: "monotone",
          },
          // 7
          {
            label: "",
            data: [0, 20, 20, 60, 60, 120, 15, 180, 120, 125, 105, 110, 170],
            borderColor: "rgb(6, 214, 160,0.7)",
            fill: false,
            cubicInterpolationMode: "monotone",
          },
          // 8
          {
            label: "",
            data: [0, 20, 20, 60, 60, 120, 15, 180, 120, 125, 105, 110, 170],
            borderColor: "rgb(17, 138, 178,0.7)",
            fill: false,
            cubicInterpolationMode: "monotone",
          },
          // 9
          {
            label: "",
            data: [0, 20, 20, 60, 60, 120, 15, 180, 120, 125, 105, 110, 170],
            borderColor: "rgb(239, 71, 111,0.7)",
            fill: false,
            cubicInterpolationMode: "monotone",
          },
          // 10
          {
            label: "",
            data: [0, 20, 20, 60, 60, 120, 15, 180, 120, 125, 105, 110, 170],
            borderColor: "rgb(255, 209, 102,0.7)",
            fill: false,
            cubicInterpolationMode: "monotone",
          },
          // 11
          {
            label: "",
            data: [0, 20, 20, 60, 60, 120, 15, 180, 120, 125, 105, 110, 170],
            borderColor: "rgb(6, 214, 160,0.7)",
            fill: false,
            cubicInterpolationMode: "monotone",
          },
          // 12
          {
            label: "",
            data: [0, 20, 20, 60, 60, 120, 15, 180, 120, 125, 105, 110, 170],
            borderColor: "rgb(17, 138, 178,0.7)",
            fill: false,
            cubicInterpolationMode: "monotone",
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          title: {
            display: true,
            text: "Chart.js Line Chart - Cubic interpolation mode",
          },
        },
        interaction: {
          intersect: false,
        },
        scales: {
          x: {
            display: true,
            title: {
              display: true,
            },
          },
          y: {
            display: true,
            title: {
              display: true,
              text: "Value",
            },
            suggestedMin: -10,
            suggestedMax: 200,
          },
        },
      },
    });
  },
};
</script>

<style lang="scss" scoped>
button:hover {
  box-shadow: none !important;
}

#chart_rada_year_hour {
  width: 916px !important;
  height: 600px !important;
}

.classification {
  line-height: 25px;
  font-size: 14px;
  color: #646464;
  background-color: #eff2fa;
  padding-top: 70px;
  padding-bottom: 70px;
  height: 100%;

  .data-vehicle-intro {
    cursor: pointer;

    h4 {
      font-size: 32px;
      color: #ea4141;
      font-size: bold;
    }

    h6 {
      font-size: 20px;
      color: #0abb87;
    }
  }
}

.card {
  border-radius: 2px;
  margin-bottom: 30px;
  border: none;
  position: relative;
  box-shadow: 0 3px 15px rgb(0 0 0 / 5%);
}
</style>