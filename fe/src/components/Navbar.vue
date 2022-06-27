<template>
  <header :class="{ 'scrolled-nav': scrollNav }">
    <nav>
      <div class="branding">
        <img src="../assets/logo.svg" alt="" />
      </div>
      <ul v-show="!mobile" class="navigation">
        <li
          v-for="nav in navbars"
          :key="nav.id"
          :class="nav.active ? 'nav-active' : ''"
          @click="navToggleClick(nav)"
        >
          <router-link class="link" :to="{ name: nav.name }">{{
            nav.name
          }}</router-link>
        </li>
      </ul>
      <div class="icon">
        <font-awesome-icon
          class="icon-font-awesome"
          @click="toggleMobileNav"
          v-show="mobile"
          :class="{ 'icon-active': mobileNav }"
          icon="fas fa-align-justify"
        />
      </div>
      <transition name="mobile-nav">
        <ul v-show="mobileNav" class="dropdown-nav">
          <li>
            <router-link class="link" :to="{ name: 'home' }">Home</router-link>
          </li>
          <li>
            <router-link class="link" :to="{ name: 'about' }"
              >About</router-link
            >
          </li>
          <li>
            <router-link class="link" :to="{ name: '' }">Porfolio</router-link>
          </li>
          <li>
            <router-link class="link" :to="{ name: 'contact' }"
              >Contact</router-link
            >
          </li>
        </ul>
      </transition>
    </nav>
  </header>
</template>

<script>
export default {
  name: "Navbar",
  created() {
    window.addEventListener("resize", this.checkScreen);
    this.checkScreen();
  },
  data() {
    return {
      scrollNav: null,
      mobile: null,
      mobileNav: null,
      windownWidth: null,
      navbars: [
        {
          id: 1,
          name: "introduce",
          active: true,
        },
        {
          id: 2,
          name: "home",
          active: false,
        },
        {
          id: 3,
          name: "classification",
          active: false,
        },
      ],

      data: [
        {
          id: 1,
          title: "Item 1",
          categories: [
            {
              id: 4,
              title: "category 4",
              description: "",
            },
            {
              id: 5,
              title: "category 5",
              description: "",
            },
          ],
        },
        {
          id: 2,
          title: "Item 2",
          categories: [],
        },
        {
          id: 3,
          title: "Item 3",
          categories: [],
        },
      ],
      car: {
        category: {
          id: 5,
          title: "category 5",
          description: "",
        },
        title: "Test",
      },
    };
  },
  mounted() {
    window.addEventListener("scroll", this.updateScroll);
  },
  methods: {
    toggleMobileNav() {
      this.mobileNav = !this.mobileNav;
    },
    updateScroll() {
      const scrollPosition = window.scrollY;
      if (scrollPosition > 50) {
        this.scrollNav = true;
        return;
      }
      this.scrollNav = false;
    },
    checkScreen() {
      this.windownWidth = window.innerWidth;
      if (this.windownWidth <= 750) {
        this.mobile = true;
        return;
      }
      this.mobile = false;
      this.mobileNav = false;
      return;
    },
    navToggleClick(nav) {
      const result = this.navbars.filter(function (elem) {
        if (elem.active == true) return elem;
      });
      result.forEach((value) => {
        value.active = false
      });
      nav.active = true
    },
  },
};
</script>

<style lang="scss" scoped>
header {
  background-color: rgba(0, 0, 0, 0.8);
  z-index: 99;
  width: 100%;
  position: fixed;
  transition: 0.5 ease all;
  color: #fff;
  nav {
    position: relative;
    display: flex;
    flex-direction: row;
    padding: 12px 0px;
    width: 90%;
    margin: 0 auto;
    @media (min-width: 1140px) {
      max-width: 1140px;
    }
    ul,
    .link {
      font-weight: 500;
      color: #fff;
      list-style: none;
      text-decoration: none;
    }
    li {
      text-transform: uppercase;
      padding: 16px;
      margin-left: 16px;
    }
    .link {
      font-size: 14px;
      transition: 0.5 ease all;
      padding-bottom: 4px;
      border-bottom: 1px solid transparent;
      &:hover {
        color: #00afea;
        border-color: #00afea;
      }
    }
    .branding {
      display: flex;
      align-items: center;
      img {
        width: 50px;
        transition: 0.5s ease all;
      }
    }
    .navigation {
      display: flex;
      align-items: center;
      flex: 1;
      justify-content: flex-end;
      .nav-active {
        background-color: #00897b !important;
      }
    }
    .icon {
      display: flex;
      align-items: center;
      position: absolute;
      top: 0;
      right: 24px;
      height: 100%;
      cursor: pointer;

      .icon-font-awesome {
        font-size: 24px;
        transition: 0.8s ease all;
      }
    }
    .icon-active {
      transform: rotate(180deg);
    }
    .dropdown-nav {
      display: flex;
      flex-direction: column;
      position: fixed;
      width: 100%;
      max-width: 250px;
      height: 100%;
      background-color: #000;
      top: 0;
      left: 0;

      li {
        margin-left: 0px;
        .link {
          color: #fff;
        }
      }
    }
    .mobile-nav-enter-active,
    .mobile-nav-leave-active {
      transition: 1s ease all;
    }
    .mobile-nav-enter-from {
      transform: translateX(-250px);
    }
    .mobile-nav-enter-to {
      transform: translateX(0);
    }
  }
}
.scrolled-nav {
  background-color: #000;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1),
    0 2px 4px -1px rgba(0, 0, 0, 0.06);

  nav {
    padding: 8px 0;
    .branding {
      img {
        width: 40px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1),
          0 2px 4px -1px rgba(0, 0, 0, 0.06);
      }
    }
  }
}
</style>
