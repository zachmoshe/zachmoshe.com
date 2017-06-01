---
layout: null
---
socialLinkEvent = function(socialNetwork) {
  ga('send', 'event', 'SocialLink', 'click', socialNetwork)
}
promotedPostEvent = function(postTitle) {
  ga('send', 'event', 'PromotedPost', 'click', postTitle)
}

$(document).ready(function () {
  $('a.blog-button').click(function (e) {
    if ($('.panel-cover').hasClass('panel-cover--collapsed')) return
    currentWidth = $('.panel-cover').width()
    if (currentWidth < 960) {
      $('.panel-cover').addClass('panel-cover--collapsed')
      $('.content-wrapper').addClass('animated slideInRight')
    } else {
      $('.panel-cover').css('max-width', currentWidth)
      $('.panel-cover').animate({'max-width': '530px', 'width': '40%'}, 400, swing = 'swing', function () {})
    }
  })

  if (window.location.hash && window.location.hash == '#blog') {
    $('.panel-cover').addClass('panel-cover--collapsed')
  }

  if (window.location.pathname !== '{{ site.baseurl }}' && window.location.pathname !== '{{ site.baseurl }}index.html') {
    $('.panel-cover').addClass('panel-cover--collapsed')
  }

  $('.btn-mobile-menu').click(function () {
    $('.navigation-wrapper').toggleClass('visible animated bounceInDown')
    $('.btn-mobile-menu__icon').toggleClass('icon-list icon-x-circle animated fadeIn')
  })

  $('.navigation-wrapper .blog-button').click(function () {
    $('.navigation-wrapper').toggleClass('visible')
    $('.btn-mobile-menu__icon').toggleClass('icon-list icon-x-circle animated fadeIn')
  })

  // activate sliders if any
  $(".bxslider").bxSlider({
      mode: 'fade',
      infiniteLoop: false,
      hideControlOnEnd: true,
      responsive: true
  })

  // hide all math after .showmath
  $(".showmath").next("p,mathpart").find("mathpart").hide();
  $(".showmath").click(function() {
      $(this).next("p,mathpart").find("mathpart").slideToggle();
  })


})
